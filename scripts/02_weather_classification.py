import os
import re
import csv
import torch
import clip
import shutil
import argparse
from PIL import Image


def classify_image(image_path, model, preprocess, text, device):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs[0]  # [sunny_prob, cloudy_prob]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_metadata(filename):
    pattern = r"(Front|Back)_(\d+\.\d+)_(\d+\.\d+)\.jpg"
    match = re.match(pattern, filename)
    if match:
        direction = match.group(1)
        lon = float(match.group(2))
        lat = float(match.group(3))
        return lon, lat, direction
    return None, None, None

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    labels = args.prompts
    text = clip.tokenize(labels).to(device)

    os.makedirs(args.sunny_dir, exist_ok=True)
    os.makedirs(args.cloudy_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    results = []
    sunny_records = [] 
    cloudy_records = []  

    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for filename in image_files:
        image_path = os.path.join(args.input_dir, filename)
        probs = classify_image(image_path, model, preprocess, text, device)
        if probs is None:
            continue

        sunny_prob, cloudy_prob = probs
        label = "Sunny" if sunny_prob >= args.threshold else "Cloudy"
        target_dir = args.sunny_dir if label == "Sunny" else args.cloudy_dir
        shutil.copy(image_path, os.path.join(target_dir, filename))

        lon, lat, direction = extract_metadata(filename)
        if lon is not None and lat is not None:
            results.append([lon, lat, direction, label])
            if label == "Sunny":
                sunny_records.append([lon, lat, direction])  
            else:
                cloudy_records.append([lon, lat, direction])  

        print(f"Classifying {filename}: {label} (Sunny: {sunny_prob:.4f}, Cloudy: {cloudy_prob:.4f})")

    # Save the main CSV
    with open(args.output_csv, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["longitude", "latitude", "image_direction", "weather"])
        writer.writerows(results)
    print(f"Results saved to {args.output_csv}")

    # Save Sunny.csv and Cloudy.csv to 'data' folder
    os.makedirs("data", exist_ok=True)

    sunny_csv_path = os.path.join("data", "Sunny.csv")
    with open(sunny_csv_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["longitude", "latitude", "image_direction"])
        writer.writerows(sunny_records)
    print(f"Sunny.csv saved to {sunny_csv_path}")

    cloudy_csv_path = os.path.join("data", "Cloudy.csv")
    with open(cloudy_csv_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["longitude", "latitude", "image_direction"])
        writer.writerows(cloudy_records)
    print(f"Cloudy.csv saved to {cloudy_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify GSV images as Sunny or Cloudy using CLIP")
    parser.add_argument("--input_dir", required=True, help="Input image folder")
    parser.add_argument("--sunny_dir", required=True, help="Output folder for sunny images")
    parser.add_argument("--cloudy_dir", required=True, help="Output folder for cloudy images")
    parser.add_argument("--output_csv", default="data/weather.csv", help="Path to output CSV file")
    parser.add_argument("--threshold", type=float, default=0.7153, help="Sunny classification threshold")
    parser.add_argument(
        "--prompts",
        nargs=2,
        default=["A photo of a sunny day", "A photo of a cloudy day"],
        help="Two prompts for CLIP classification"
    )

    args = parser.parse_args()
    main(args)
