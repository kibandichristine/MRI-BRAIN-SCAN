import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = r'C:\Users\KibandiChristineWamb\Downloads\MRI BRAIN SCAN\combined_images'
OUTPUT_CSV = r'C:\Users\KibandiChristineWamb\Downloads\MRI BRAIN SCAN\All_Disease1.csv'

# Label to filename prefix mapping
LABELS = {
    'MildDemented': 'Mild_D',
    'ModerateDemented': 'Mod_D',
    'NonDemented': 'Non_D',
    'VeryMildDemented': 'VMild_D'
}

# Initialize counters for standardized filenames
counters = {label: 1 for label in LABELS}

# === FEATURE EXTRACTION FUNCTION ===
def extract_features(img_path):
    try:
        img = Image.open(img_path).convert('L')  # Grayscale
        img_np = np.array(img)

        mean_pixel = np.mean(img_np)
        std_pixel = np.std(img_np)
        entropy = shannon_entropy(img_np)
        edges = sobel(img_np)
        edge_density = np.mean(edges)

        h, w = img_np.shape
        ch, cw = h // 4, w // 4
        center = img_np[h//2 - ch//2:h//2 + ch//2, w//2 - cw//2:w//2 + cw//2]
        center_brightness = np.mean(center)

        return mean_pixel, std_pixel, entropy, edge_density, center_brightness

    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        return None

# === MAIN LOOP ===
data = []

print("🔍 Starting feature extraction...")

for label, prefix in LABELS.items():
    folder_path = os.path.join(BASE_DIR, label)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    for file in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, file)
            features = extract_features(full_path)

            if features:
                count = counters[label]
                standardized_name = f"{prefix}_{count:05d}"  # e.g. Mild_D_00001
                counters[label] += 1

                is_aug = 'TRUE' if 'aug' in file.lower() else 'FALSE'

                data.append([
                    standardized_name,
                    label,
                    *features,
                    is_aug
                ])

# === SAVE TO CSV ===
df = pd.DataFrame(data, columns=[
    'Filename', 'Label',
    'mean_pixel_intensity', 'std_pixel_intensity',
    'entropy', 'edge_density', 'center_brightness',
    'is_augmented'
])

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done! CSV saved at: {OUTPUT_CSV}")
