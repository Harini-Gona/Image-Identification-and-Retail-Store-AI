import os
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_FOLDER = "data/images"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)

embeddings = []
image_names = []

print("Processing images...")

for img_name in tqdm(os.listdir(IMAGE_FOLDER)):
    try:
        path = os.path.join(IMAGE_FOLDER, img_name)
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy()[0])
        image_names.append(img_name)

    except:
        continue

np.save("embeddings.npy", np.array(embeddings))
np.save("image_names.npy", np.array(image_names))

print("✅ Done! Embeddings saved.")