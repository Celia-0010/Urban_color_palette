import os
import json
import random
import numpy as np
import torch
import pandas as pd
import colorsys
import gc
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import argparse

# Set multiprocessing start method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Category ID mapping
CATEGORY_MAP = {
    "building": 1, "road": 2, "sidewalk": 3, "overpass": 4, "tunnel": 5,
    "sky": 6, "tree": 7, "grass": 8, "shrub": 9, "flower bed": 10, "rock": 11,
    "soil": 12, "mountain": 13, "sea": 14, "awning": 15, "pole": 16, "fence": 17,
    "bench": 18, "dustbin": 19, "kiosk": 20, "streetlight": 21, "traffic light": 22,
    "traffic cone": 23, "traffic barrier": 24, "billboard": 25, "store sign": 26,
    "signpost": 27, "traffic sign": 28, "pedestrian": 29, "motorcycle": 30,
    "car": 31, "bus": 32, "tram": 33, "truck": 34
}
CAT_IDS = list(CATEGORY_MAP.values())

def load_kmeans_model(model_path):
    model = torch.load(model_path)
    return model['centers'], model['hyperparameters']['K']

def get_color_info(centers):
    color_info = {}
    for i, center in enumerate(centers):
        center = center.cpu().numpy() if isinstance(center, torch.Tensor) else center
        r, g, b = center
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        h = h * 360
        color_info[f"center{i}"] = {
            "RGB": center.tolist(),
            "HLS": (float(h), float(l), float(s))
        }
    return color_info

def get_image_mask_pairs(img_base, mask_base):
    pairs = []
    image_files = [f for f in os.listdir(img_base) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in image_files:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_base, img_name)
        mask_path = os.path.join(mask_base, f"mask_{base}.png")
        
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    
    return pairs


def classify_pixels_gpu(pixels, centers_gpu):
    if pixels.size == 0:
        return np.array([])
    pixels_norm = pixels.reshape(-1, 3) / 255.0
    pixels_tensor = torch.tensor(pixels_norm, device=centers_gpu.device).float()
    centers_norm = centers_gpu.float()
    dist = torch.norm(pixels_tensor[:, None] - centers_norm, dim=2)
    return torch.argmin(dist, dim=1).cpu().numpy()

def process_image(img_path, mask_path, centers, K, device):
    try:
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        if img.shape[:2] != mask.shape:
            return None
    except Exception:
        return None

    parts = os.path.splitext(os.path.basename(img_path))[0].split('_')
    if len(parts) != 3:
        return None
    direction, lon, lat = parts

    classifications = {}
    all_pixels = []
    centers_gpu = centers.clone().detach().to(device)

    for cat_id in CAT_IDS:
        mask_val = cat_id * 7
        pixels = img[np.where(mask == mask_val)]
        if pixels.size == 0:
            classifications[cat_id] = np.array([])
            continue
        all_pixels.append(pixels)
        classifications[cat_id] = classify_pixels_gpu(pixels, centers_gpu)

    total_pixels = sum(len(p) for p in all_pixels)
    row = {'longitude': lon, 'latitude': lat, 'image_direction': direction}

    for cat_id in CAT_IDS:
        labels = classifications[cat_id]
        counts = np.bincount(labels, minlength=K) if labels.size else np.zeros(K)
        for cid in range(K):
            row[f'id{cat_id}_center{cid}'] = counts[cid] / total_pixels if total_pixels else 0.0

    return row

def process_batch(batch, centers, K, device):
    results = []
    for img_path, mask_path in batch:
        row = process_image(img_path, mask_path, centers, K, device)
        if row:
            results.append(row)
    return results

def main_process(img_base, mask_base, model_path, output_parquet, color_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    centers, K = load_kmeans_model(model_path)

    with open(color_json, 'w') as f:
        json.dump(get_color_info(centers), f, indent=4)

    pairs = get_image_mask_pairs(img_base, mask_base)
    random.shuffle(pairs)

    pool = Pool(processes=15)
    batch_size = 30
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    all_results = []
    for batch in tqdm(batches, desc="Processing"):
        batch_results = pool.apply(process_batch, (batch, centers, K, device))
        all_results.extend(batch_results)

    pool.close()
    pool.join()

    df = pd.DataFrame(all_results)
    cols = ['longitude', 'latitude', 'image_direction'] + [f'id{t}_center{c}' for t in CAT_IDS for c in range(K)]
    df[cols].to_parquet(output_parquet, engine='pyarrow')

    del df
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract color distributions from semantic masks using incremental k-means clusters.")
    parser.add_argument('--image_base', type=str, required=True, help="Folder with RGB images named as <direction>_<lon>_<lat>.jpg")
    parser.add_argument('--mask_base', type=str, required=True, help="Folder with semantic masks named as mask_<image_name>.png")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved k-means model (.pth)")
    parser.add_argument('--output_parquet', type=str, required=True, help="Output .parquet file path for pixel ratio table")
    parser.add_argument('--color_json', type=str, required=True, help="Output .json file for cluster color info")
    args = parser.parse_args()

    main_process(
        img_base=args.image_base,
        mask_base=args.mask_base,
        model_path=args.model_path,
        output_parquet=args.output_parquet,
        color_json=args.color_json
    )

    