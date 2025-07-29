import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import time
import argparse
from mpl_toolkits.mplot3d import Axes3D

def parse_args():
    parser = argparse.ArgumentParser(description="Incremental K-Means clustering on masked image pixels")
    parser.add_argument(
        "--image_base", type=str, required=True,
        help="Folder with RGB images named as <direction>_<lon>_<lat>.jpg"
    )
    parser.add_argument(
        "--mask_base", type=str, required=True,
        help="Folder with semantic masks named as mask_<image_name>.png"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="Directory where outputs (plots, palette, model) will be saved."
    )
    parser.add_argument(
        "--K", type=int, default=50,
        help="Number of clusters (color centers)."
    )
    parser.add_argument(
        "--sample_images", type=int, default=None,
        help="Number of image–mask pairs to sample (default: all)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="Number of image batches to accumulate before processing."
    )
    parser.add_argument(
        "--check_interval", type=int, default=20,
        help="Interval (in batches) at which to record convergence."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def stream_pixels(image_base, mask_base, sample_images, batch_size):
    """Generate batches of valid pixels from masked images."""
    image_mask_pairs = []

    # List all RGB image files in the image_base folder
    for img_name in os.listdir(image_base):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(image_base, img_name)
            mask_name = f"mask_{os.path.splitext(img_name)[0]}.png"
            mask_path = os.path.join(mask_base, mask_name)
            if os.path.exists(mask_path):
                image_mask_pairs.append((img_path, mask_path))

    if not image_mask_pairs:
        raise ValueError("No valid image–mask pairs found in the specified folders.")

    # Optionally sample a subset
    if sample_images and sample_images < len(image_mask_pairs):
        image_mask_pairs = random.sample(image_mask_pairs, sample_images)
    else:
        random.shuffle(image_mask_pairs)

    temp_batches = []
    for img_path, mask_path in image_mask_pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        valid_pixels = img_rgb[mask > 0]
        if valid_pixels.size == 0:
            continue

        batch_tensor = torch.from_numpy(valid_pixels).float() / 255.0
        temp_batches.append(batch_tensor)
        print(f"Processed {os.path.basename(img_path)}, extracted {valid_pixels.shape[0]} valid pixels")

        if len(temp_batches) == batch_size:
            yield torch.cat(temp_batches, dim=0)
            temp_batches = []

    if temp_batches:
        yield torch.cat(temp_batches, dim=0)

def incremental_kmeans(stream, K, device, check_interval, output_folder):
    """Perform incremental K-Means clustering with convergence tracking."""
    centers = None
    counts = torch.zeros(K, device=device)
    first_batch = True
    batch_counter = 0
    diff_history = []
    prev_centers = None

    plt.figure(figsize=(8, 6))
    plt.xlabel(f"Batch Checkpoints (every {check_interval} batches)")
    plt.ylabel("Maximum Center Change (L2 Distance)")
    plt.title("Cluster Center Convergence Tracking")

    for batch in stream:
        batch = batch.to(device)
        if first_batch:
            if batch.size(0) < K:
                continue
            perm = torch.randperm(batch.size(0))
            centers = batch[perm[:K]].clone()
            counts += 1
            prev_centers = centers.clone()
            first_batch = False
            continue

        distances = torch.cdist(batch, centers)
        labels = torch.argmin(distances, dim=1)

        for j in range(K):
            mask = (labels == j)
            if mask.any():
                n_j = mask.sum().float()
                sum_j = batch[mask].sum(dim=0)
                centers[j] = (centers[j] * counts[j] + sum_j) / (counts[j] + n_j)
                counts[j] += n_j

        batch_counter += 1
        if batch_counter % check_interval == 0:
            with torch.no_grad():
                diffs = torch.norm(centers - prev_centers, dim=1)
                max_diff = diffs.max().item()
                diff_history.append(max_diff)
                print(f"After {batch_counter} batches: max center change = {max_diff:.6f}")

                x = [i * check_interval for i in range(1, len(diff_history)+1)]
                plt.plot(x, diff_history, 'b-')
                plt.xlabel(f"Batch Checkpoints (batch size: {stream.__defaults__[0]})")
                plt.ylabel("Maximum Center Change (L2 Distance)")
                plt.title("Cluster Center Convergence Tracking")
                plt.savefig(os.path.join(output_folder, "convergence_curve.png"), dpi=300)
                plt.clf()

            prev_centers = centers.clone()

    if diff_history:
        plt.figure(figsize=(8, 6))
        x = [i * check_interval for i in range(1, len(diff_history)+1)]
        plt.plot(x, diff_history, color='black', linewidth=1)
        plt.ylim(0, max(diff_history) * 1.1)
        plt.xlabel(f"Batch Checkpoints (batch size: {stream.__defaults__[0]})")
        plt.ylabel("Maximum Center Change (L2 Distance)")
        plt.title("Final Incremental K-Means Convergence Curve")
        plt.savefig(os.path.join(output_folder, "final_convergence.png"), dpi=300)
        plt.close()

    return centers, counts, diff_history

def main():
    args = parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()
    pixel_stream = stream_pixels(
        args.image_base,
        args.mask_base,
        args.sample_images,
        args.batch_size
    )
    centers, counts, diff_history = incremental_kmeans(
        pixel_stream,
        args.K,
        device,
        args.check_interval,
        args.output_folder
    )
    total_time = time.time() - start_time

    model_state = {
        'centers': centers.cpu(),
        'counts': counts.cpu(),
        'diff_history': diff_history,
        'hyperparameters': {
            'K': args.K,
            'sample_images': args.sample_images,
            'batch_size': args.batch_size,
            'check_interval': args.check_interval,
            'seed': args.seed
        },
        'total_time': total_time
    }
    save_path = os.path.join(args.output_folder, "kmeans_model_full.pth")
    torch.save(model_state, save_path)
    print(f"Model saved to {save_path}")

    # generate palette
    centers_np = centers.cpu().numpy()
    sorted_idx = np.lexsort((centers_np[:, 2], centers_np[:, 1], centers_np[:, 0]))
    sorted_centers = centers_np[sorted_idx]

    palette = np.zeros((50, 800, 3))
    block_width = 800 // args.K
    for i in range(args.K):
        start = i * block_width
        end = start + block_width
        palette[:, start:end] = sorted_centers[i]

    plt.figure(figsize=(12, 2))
    plt.imshow(palette)
    plt.axis('off')
    plt.title(f"Color Palette (K={args.K})")
    palette_path = os.path.join(args.output_folder, "color_palette.png")
    plt.savefig(palette_path, dpi=300)
    plt.close()
    print(f"Color palette saved to {palette_path}")

if __name__ == "__main__":
    main()
