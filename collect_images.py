import os
import shutil
import random

# SOURCE (your downloaded dataset)
SOURCE = r"C:\Users\armys\Downloads\archive\img_highres"

# DESTINATION (your project folder)
DEST = r"C:\Users\armys\OneDrive\Documents\Desktop\ai-retail\data\images"

# Create destination if not exists
os.makedirs(DEST, exist_ok=True)

all_images = []

for root, dirs, files in os.walk(SOURCE):
    for file in files:
        if file.endswith(".jpg"):
            full_path = os.path.join(root, file)
            all_images.append(full_path)

print(f"Total images found: {len(all_images)}")


selected = random.sample(all_images, 5000)


for i, img_path in enumerate(selected):
    new_name = f"img_{i}.jpg"
    shutil.copy(img_path, os.path.join(DEST, new_name))

print("✅ Done copying images!")