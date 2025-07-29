import argparse
import os
import copy
import time
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import threading
import concurrent.futures
import queue
import random
from collections import defaultdict

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, SamPredictor 
import cv2
import matplotlib.pyplot as plt

def load_image(image_path, device):
    """Load and preprocess the image."""
    image_pil = Image.open(image_path).convert("RGB")  # Load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    image = image.to(device)  # Move image tensor to device
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    """Load the Grounding DINO model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model load result:", load_res)
    model.eval().to(device)  # Ensure model is on the correct device
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """Get grounding outputs from the model."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    # Ensure model and image are on the correct device
    model = model.to(device)
    image = image.to(device)

    with torch.inference_mode():  # Use inference_mode for better performance
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # Keep on device
    boxes = outputs["pred_boxes"][0]  # Keep on device

    # Filter output
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # num_filt, 256
    boxes_filt = boxes[filt_mask]  # num_filt, 4

    # Get phrase
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # Build predictions
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    """Draw a segmentation mask on the image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    """Draw bounding boxes and their labels on the image."""
    box = box.cpu().numpy()
    x0, y0 = box[0].item(), box[1].item() 
    w, h = box[2].item() - box[0].item(), box[3].item() - box[1].item()
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def run_segmentation(
    text_prompt,
    boxes_filt,
    pred_phrases,
    predictor,
    image_cv2,
    device
):
    """Run SAM to generate masks based on bounding boxes."""
    if boxes_filt.numel() == 0:
        print(f"Object not detected for prompt: '{text_prompt}'")
        return None, None  # Return None for later processing
    else:
        # Apply box transformations and ensure on device
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv2.shape[:2]).to(device)
        
        if transformed_boxes.numel() == 0:
            print("Transformed boxes are empty after applying transformation.")
            return None, None

        # Perform segmentation prediction
        with torch.inference_mode():  # Ensure no gradient computation
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        return masks, pred_phrases

def ensure_dir(path):
    """Ensure that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

class ImageLoader:
    def __init__(self, image_files, images_folder, device, max_queue_size=10):
        """
        Initializes the ImageLoader with image files and starts the background thread.
        
        Args:
            image_files (list): List of image filenames to load.
            images_folder (str): Path to the folder containing images.
            device (torch.device): The device to which tensors are moved.
            max_queue_size (int): Maximum number of images to preload.
        """
        self.image_files = image_files
        self.images_folder = images_folder
        self.device = device
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._stop_event = threading.Event()
        self.executor.submit(self._load_images)

    def _load_images(self):
        """Background thread method to load and preprocess images."""
        for image_file in self.image_files:
            if self._stop_event.is_set():
                break
            image_path = os.path.join(self.images_folder, image_file)
            try:
                # Load and preprocess the image
                image_pil, image = load_image(image_path, device=self.device)
                
                # Load image with OpenCV for SAM
                image_cv2 = cv2.imread(image_path)
                if image_cv2 is not None:
                    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
                else:
                    print(f"Error: Unable to read image at path: {image_path}")
                    continue
                
                # Put the loaded image into the queue
                self.queue.put((image_file, image_pil, image, image_cv2))
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
        # Signal that loading is done
        self.queue.put(None)

    def get_image(self):
        """
        Retrieves the next preloaded image from the queue.
        
        Returns:
            tuple or None: Returns a tuple (image_file, image_pil, image, image_cv2) or None if loading is complete.
        """
        item = self.queue.get()
        if item is None:
            return None
        return item

    def stop(self):
        """Stops the image loading thread and cleans up resources."""
        self._stop_event.set()
        self.executor.shutdown(wait=False)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grounded-SAM Semantic Segmentation Pipeline")

    parser.add_argument("--images_folder", type=str, required=True,
                        help="Folder containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output mask images")
    parser.add_argument("--config_file", type=str,
                        default="/root/grounded-sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to the Grounding DINO config file")
    parser.add_argument("--grounded_checkpoint", type=str,
                        default="/root/grounded-sam/groundingdino_swint_ogc.pth",
                        help="Path to the Grounding DINO model checkpoint")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="/root/grounded-sam/sam_vit_h_4b8939.pth",
                        help="Path to the SAM model checkpoint")

    args = parser.parse_args()

    # Fixed thresholds (not exposed to CLI)
    box_threshold = 0.3
    text_threshold = 0.25

    # Timing
    t0 = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assign paths
    config_file = args.config_file
    grounded_checkpoint = args.grounded_checkpoint
    sam_checkpoint = args.sam_checkpoint
    images_folder = args.images_folder
    output_dir = args.output_dir

    # Make sure output directory exists
    ensure_dir(output_dir)

    # Process parameters
    box_threshold = 0.3
    text_threshold = 0.25
    # Define runs for background and decoration
    runs = [
        {
            "text_prompt": "building . road . sidewalk . overpass . tunnel . sky . tree . grass . shrub . flower bed . rock . soil . mountain . sea .",
            "type": "background"
        },
        {
            "text_prompt": "awning . pole . fence . bench . dustbin . kiosk . streetlight . traffic light . traffic cone . traffic barrier . billboard . store sign . signpost . traffic sign . pedestrian . motorcycle . car . bus . tram . truck .",
            "type": "decoration"
        }
    ]
    
    # Mapping categories to values
    category_to_value = {
    "building": 1,
    "road": 2,
    "sidewalk": 3,
    "overpass": 4,
    "tunnel": 5,
    "sky": 6,
    "tree": 7,
    "grass": 8,
    "shrub": 9,
    "flower bed": 10,
    "rock": 11,
    "soil": 12,
    "mountain": 13,
    "sea": 14,
    "awning": 15,
    "pole": 16,
    "fence": 17,
    "bench": 18,
    "dustbin": 19,
    "kiosk": 20,
    "streetlight": 21,
    "traffic light": 22,
    "traffic cone": 23,
    "traffic barrier": 24,
    "billboard": 25,
    "store sign": 26,
    "signpost": 27,
    "traffic sign": 28,
    "pedestrian": 29,
    "motorcycle": 30,
    "car": 31,
    "bus": 32,
    "tram": 33,
    "truck": 34
}

    # Load model
    print("\nLoading Grounding DINO model...")
    model = load_model(config_file, grounded_checkpoint, device=device)
    print("Grounding DINO model loaded successfully.")
    t1 = time.time()

    print("\nInitializing SAM predictor...")
    sam = build_sam(checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    print("SAM predictor initialized successfully.")
    t2 = time.time()

    # Get the list of image files
    image_files = [f for f in os.listdir(images_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"\nFound {len(image_files)} images in '{images_folder}'")

    # Initialize image loader
    loader = ImageLoader(image_files, images_folder, device=device)

    # Image processing loop
    img_idx = 0
    while True:
        item = loader.get_image()
        if item is None:
            break
            
        image_file, image_pil, image, image_cv2 = item
        img_idx += 1
        image_name = os.path.splitext(image_file)[0]
        print(f"\nProcessing image {img_idx}/{len(image_files)}: {image_file}")

        # Initialize storage containers
        run_results = {
            'background': {'phrases': [], 'masks': None},
            'decoration': {'phrases': [], 'masks': None}
        }

        predictor.set_image(image_cv2)

        # Process each type
        for run in runs:
            text_prompt = run["text_prompt"]
            run_type = run["type"]
            print(f"\nProcessing {run_type}...")
            # ==================================================================
            # Stage 1: Preprocess categories
            # ==================================================================
            original_categories = [c.strip().lower() for c in text_prompt.split('.') if c.strip()]
            
            # Build category variants dictionary
            category_variants = defaultdict(list)
            for cat in original_categories:
                # Standard form
                standard = ' '.join(cat.split())
                category_variants[standard].append(cat)
                # Compact form
                compact = standard.replace(' ', '')
                if compact != standard:
                    category_variants[compact].append(cat)
                # Singular form
                if cat.endswith('s'):
                    category_variants[cat[:-1]].append(cat)
            # ==================================================================
            # Stage 2: Grounding DINO prediction
            # ==================================================================
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt,
                box_threshold, text_threshold,
                with_logits=True,
                device=device
            )
            # ==================================================================
            # Stage 3: Predict phrase correction
            # ==================================================================
            def find_best_match(query, candidates):
                """Find the best match using an improved similarity calculation"""
                # Preprocess the query term
                query = query.lower().strip()
                
                # Try direct match first
                if query in candidates:
                    return query
                
                # Split query term into word set
                query_words = set(query.split())
                max_jaccard = -1
                best_matches = []
                
                # Compute Jaccard similarity
                for candidate in candidates:
                    candidate_words = set(candidate.split())
                    intersection = query_words & candidate_words
                    union = query_words | candidate_words
                    jaccard = len(intersection) / len(union) if union else 0
                    
                    if jaccard > max_jaccard:
                        max_jaccard = jaccard
                        best_matches = [candidate]
                    elif jaccard == max_jaccard:
                        best_matches.append(candidate)
                
                # Return the best match (randomly choose in case of multiple matches)
                return random.choice(best_matches) if best_matches else query

            accurate_pred_phrases = []
            for phrase in pred_phrases:
                # Extract base phrase and logit
                base_phrase = phrase.split('(')[0].strip().lower()
                logit_suffix = phrase.split('(')[1] if '(' in phrase else ''
                
                # Match the best category
                matched_category = find_best_match(base_phrase, original_categories)
                
                # Handle compound word special cases
                if ' ' in matched_category and matched_category.replace(' ', '') in category_variants:
                    final_category = category_variants[matched_category.replace(' ', '')]
                else:
                    final_category = category_variants.get(matched_category, matched_category)
                
                # Rebuild the corrected phrase
                accurate_phrase = f"{final_category}({logit_suffix}" if logit_suffix else final_category
                accurate_pred_phrases.append(accurate_phrase)

            print(f"Original {run_type} pred_phrases: {pred_phrases}")
            print(f"Accurate {run_type} pred_phrases: {accurate_pred_phrases}")
            # ==================================================================
            # Stage 4: Bounding box adjustment
            # ==================================================================
            W, H = image_pil.size
            scaling_tensor = torch.tensor([W, H, W, H], device=device)
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * scaling_tensor
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            # ==================================================================
            # Stage 5: SAM segmentation
            # ==================================================================
            masks, _ = run_segmentation(
                text_prompt=text_prompt,
                boxes_filt=boxes_filt,
                pred_phrases=accurate_pred_phrases,
                predictor=predictor,
                image_cv2=image_cv2,
                device=device
            )
            print(f"Initial {run_type} masks shape: {masks.shape}")
            # ==================================================================
            # Stage 6: Pixel-level correction and merging
            # ==================================================================
            if masks is not None and len(masks) > 0:
                
                # Extract logits and convert to tensor
                logits = []
                for phrase in accurate_pred_phrases:
                    if '(' in phrase:
                        logit_str = phrase.split('(')[1].split(')')[0].strip()
                        logits.append(float(logit_str))
                    else:
                        logits.append(0.0)
                logits_tensor = torch.tensor(logits, device=device)

                # Optimized pixel-level correction (using matrix operations)
                masks_flat = masks.view(masks.size(0), -1)  # [N, H*W]
                logits_flat = logits_tensor.view(-1, 1).expand_as(masks_flat)

                # Calculate pixel conflict mask --------------------------------------------------
                sum_per_pixel = masks_flat.sum(dim=0)  # Number of objects per pixel
                conflict_mask = sum_per_pixel > 1      # Pixels that need correction

                # Find the index of the maximum logit for each pixel
                max_indices = torch.argmax(logits_flat * masks_flat.float(), dim=0)

                # Generate corrected mask (only modify conflicted pixels)-----------------------------------------
                revised_masks = masks_flat.clone()  # Keep original mask

                if conflict_mask.any():
                    # Create one-hot encoding of the max index (only process conflicting pixels)
                    rows = max_indices[conflict_mask]
                    cols = torch.arange(masks_flat.size(1), device=device)[conflict_mask]
                    
                    # Clear original values for conflicting pixels
                    revised_masks[:, conflict_mask] = False  
                    # Set new max values
                    revised_masks[rows, cols] = True

                revised_masks = revised_masks.view_as(masks)
                masks = revised_masks.bool()
                # ==========================================================================================
                # Final merging stage (optimized version)
                # ------------------------------------------------------------------------------------------
                # Create phrase to index mapping
                phrase_groups = defaultdict(list)
                for idx, phrase in enumerate(accurate_pred_phrases):
                    clean_phrase = phrase.split('(')[0].strip()
                    phrase_groups[clean_phrase].append(idx)
                
                # Merge masks
                merged_masks = []
                merged_phrases = []
                for phrase, indices in phrase_groups.items():
                    combined_mask = torch.any(masks[indices, ...], dim=0).unsqueeze(0)
                    merged_masks.append(combined_mask)
                    merged_phrases.append(phrase)
                
                # Update final result
                if merged_masks:
                    masks = torch.cat(merged_masks, dim=0)
                    pred_phrases = merged_phrases

                # Save current run results
                run_results[run_type]['phrases'] = merged_phrases
                run_results[run_type]['masks'] = masks
                print(f"Final {run_type} merged phrases: {merged_phrases}")
                print(f"Final {run_type} masks shape: {masks.shape}")
                        
            # Memory cleanup
            del boxes_filt, masks
            torch.cuda.empty_cache()
        # ==================================================================
        # Stage 7: Cross-type result merging
        # ==================================================================
        def combine_results(bg_result, deco_result):
            """Merge background and decoration results, decoration takes precedence"""
            bg_masks = bg_result['masks']
            deco_masks = deco_result['masks']
            
            # Handle empty result cases
            if bg_masks is None and deco_masks is None:
                return None
            if bg_masks is None:
                return deco_result
            if deco_masks is None:
                return bg_result
            # Concatenate results
            total_phrases = bg_result['phrases'] + deco_result['phrases']
            total_masks = torch.cat([bg_masks, deco_masks], dim=0)
            # Create decoration area mask
            deco_start = len(bg_result['phrases'])
            deco_area = total_masks[deco_start:].any(dim=0)

            # Generate final mask
            final_masks = []
            for i in range(total_masks.size(0)):
                mask = total_masks[i].clone()
                if i < deco_start:  # Background categories should exclude decoration area
                    mask[deco_area] = False
                final_masks.append(mask)
            
            return {
                'phrases': total_phrases,
                'masks': torch.stack(final_masks)
            }

        final_result = combine_results(
            run_results['background'],
            run_results['decoration']
        )
        print(f"\nFinal_result['masks'].shape: {final_result['masks'].shape}")
        print(f"Final_result['phrases']: {final_result['phrases']}")
        # print(f"final_mask min: {final_result['masks'].min()}, max: {final_result['masks'].max()}") 
        
        
        # ==================================================================
        # Stage 8: Map category names to values and generate the final matrix
        # ==================================================================
        def generate_final_matrix(final_result, category_to_value, image_name):
            # Get the mask tensor and category names
            masks = final_result['masks']  # shape: [num_categories, 1, 640, 640]
            phrases = final_result['phrases']  # ['sky', 'road', ...]
            
            # Create an empty 2D matrix with shape [640, 640], initialized to 0
            final_matrix = torch.zeros((640, 640), dtype=torch.int32)
            
            # Iterate through each category
            for i, phrase in enumerate(phrases):
                # Get category value
                # Remove the logit value from parentheses and get the pure category name
                clean_phrase = phrase.strip("[]").strip("'").strip()  # Convert ['sky'] to "sky"
                
                # Get the corresponding value, default to 0 if category does not exist
                category_value = category_to_value.get(clean_phrase, 0)
                
                # Get the corresponding mask for that category
                mask = masks[i, 0, :, :]  # Each category has one mask, shape [H, W]
                
                # Fill the corresponding mask area with the category value
                final_matrix[mask.bool()] = category_value
            
            print(f"Final matrix shape: {final_matrix.shape}")
            print(f"Final matrix: \n{final_matrix}")
            
            # Multiply the final matrix by 7 to increase the gap between categories
            final_matrix = final_matrix * 7

            # Convert the final PyTorch tensor to a NumPy array
            final_matrix = final_matrix.numpy()

            # Convert the NumPy array to an Image object, ensuring pixel values are between 0-255
            final_matrix_image = Image.fromarray(final_matrix.astype(np.uint8))

            # Save as a grayscale image, with the filename mask_original_image_name.png
            final_matrix_filename = os.path.join(output_dir, f"mask_{image_name}.png")
            final_matrix_image.save(final_matrix_filename)
            
            print(f"Saved mask image as mask_{image_name}.png")
            return final_matrix
        
        final_matrix = generate_final_matrix(final_result, category_to_value, image_name)
        # Final cleanup
        del image_pil, image, image_cv2
        if final_result is not None:
            del final_result['masks']
        torch.cuda.empty_cache()
    # Stop loader
    loader.stop()
    # Performance statistics
    t3 = time.time()
    print("\nProcessing completed.")
    print(f"Load Grounding DINO model: {t1 - t0:.2f} seconds")
    print(f"Initialize SAM: {t2 - t1:.2f} seconds")
    print(f"Process {len(image_files)} images: {t3 - t2:.2f} seconds")
    print(f"Total time consuming: {t3 - t0:.2f} seconds")
