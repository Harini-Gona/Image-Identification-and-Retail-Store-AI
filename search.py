import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
import os

BASE_DIR = os.path.dirname(__file__)
# Load embeddings
embeddings = np.load(os.path.join(BASE_DIR, "embeddings.npy"))
image_names = np.load(os.path.join(BASE_DIR, "image_names.npy"))


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)


def search_image(query_path, top_k=5):
    image = preprocess(Image.open(query_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model.encode_image(image)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

    query_emb = query_emb.cpu().numpy()

    distances, indices = index.search(query_emb, top_k)
    

    results = []
    for idx in indices[0]:
        results.append(image_names[idx])

    return results


query = "data/images/img_10.jpg"   # change if needed
results = search_image(query)

print("🔍 Similar images:")
for r in results:
    print(r)