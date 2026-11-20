
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
import shutil
import os
import random
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory="data"), name="data")

try:
    embeddings = np.load("embeddings.npy")
    image_names = np.load("image_names.npy")
    print("✅ Embeddings loaded")
except Exception as e:
    print("❌ ERROR loading embeddings:", e)
    embeddings = np.random.rand(10, 512).astype("float32")
    image_names = np.array(["sample.jpg"] * 10)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)

def detect_color_simple(image_path):
    try:
        import cv2
        import numpy as np

        img = cv2.imread(image_path)

        if img is None:
            return "gray"

        h, w, _ = img.shape
        img = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixels = img.reshape(-1, 3)

      
        pixels = pixels[np.any(pixels < 240, axis=1)]

        r, g, b = pixels[:,0], pixels[:,1], pixels[:,2]
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) &
            (abs(r - g) > 15) &
            (r > g) & (r > b)
        )
        pixels = pixels[~skin_mask]

        if len(pixels) == 0:
            return "white"

        r, g, b = pixels.mean(axis=0)

        # color logic
        if r > 180 and g > 180 and b > 180:
            return "white"
        elif abs(r - g) < 20 and abs(r - b) < 20:
            return "gray"
        elif r < 60 and g < 60 and b < 60:
            return "black"
        elif r > 150 and g < 100 and b < 100:
            return "red"
        elif b > 150 and r < 120:
            return "blue"
        elif g > 150 and r < 120:
            return "green"
        elif r > 180 and b > 150 and g < 180:
            return "pink"
        elif r > 180 and g > 180 and b < 100:
            return "yellow"
        else:
            return "gray"

    except Exception as e:
        print("COLOR ERROR:", e)
        return "gray"
    
def extract_category(filename):
    name = filename.lower()
    if "dress" in name:
        return "Dress"
    elif "shirt" in name:
        return "Shirt"
    elif "jeans" in name:
        return "Jeans"
    elif "jacket" in name:
        return "Jacket"
    elif "shoe" in name:
        return "Shoes"
    else:
        return "Fashion Item"

@app.post("/search")
async def search(file: UploadFile = File(...)):
    global embeddings, image_names, index  
    filepath = "temp.jpg"
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(filepath).convert("RGB")
        image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        emb = emb.cpu().numpy().astype("float32")

        distances, indices = index.search(emb, 8)

        threshold = 0.20
        is_new = False

        if distances[0][0] > threshold:
            is_new = True
            print("⚠️ New image detected")

            new_name = f"new_{len(image_names)}.jpg"
            new_path = os.path.join("data/images", new_name)

            shutil.copy(filepath, new_path)
        
            embeddings = np.vstack([embeddings, emb])
            image_names = np.append(image_names, new_name)

            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype("float32"))

            np.save("embeddings.npy", embeddings)
            np.save("image_names.npy", image_names)

            print("✅ Added to database")

        base_url = "http://localhost:8000/data/images/"
        results = []

        for idx in indices[0]:
            img_name = image_names[idx]
            product_id = int(hashlib.md5(img_name.encode()).hexdigest(), 16) % 1000000
            category = extract_category(img_name)
            image_path = os.path.join("data/images", img_name)
            color = detect_color_simple(image_path)

            results.append({
                "id": product_id,
                "image": base_url + img_name,
                "name": category,
                "brand": "FashionHub",
                "color": color,
                "price": int(800 + random.randint(200, 1500))
            })


        if is_new:
            new_image_path = os.path.join("data/images", new_name)

            try:
                color = detect_color_simple(new_image_path)
            except Exception as e:
                print("NEW IMAGE COLOR ERROR:", e)
                color = "gray"

            product_data = {
                "id": -1,
                "image": base_url + new_name,
                "name": "New Item (Not in DB)",
                "brand": "Unknown",
                "color": color,
                "price": 999
            }

        else:
            product_data = {
    "id": int(indices[0][0]),
    "image": base_url + image_names[indices[0][0]],
    "name": extract_category(image_names[indices[0][0]]),
    "brand": "FashionHub",
    "color": detect_color_simple(filepath), 
    "price": int(800 + random.randint(200, 1500))
}

            try:
                main_img_name = image_names[indices[0][0]]
                main_path = os.path.join("data/images", main_img_name)
                product_data["color"] = detect_color_simple(main_path)
            except Exception as e:
                print("EXISTING IMAGE COLOR ERROR:", e)
                product_data["color"] = "gray"

        return {
            "results": results,
            "product": product_data,
            "new": is_new
        }

    except Exception as e:
        print("❌ ERROR in search:", e)
        return {"error": str(e)}

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)