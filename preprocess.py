import os
import cv2
import numpy as np
import random

# Define directories
ORIGINAL_FOLDER = "images"
DAMAGED_FOLDER = "damaged_images"
PROCESSED_ORIGINAL_FOLDER = "preprocessed/original"
PROCESSED_DAMAGED_FOLDER = "preprocessed/damaged"
STRUCTURE_MAP_FOLDER = "preprocessed/structure_maps"
DEBUG_FOLDER = "preprocessed/debug_structure_maps"

# Create output directories if they don't exist
os.makedirs(PROCESSED_ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_DAMAGED_FOLDER, exist_ok=True)
os.makedirs(STRUCTURE_MAP_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

IMAGE_SIZE = (256, 256)

def normalize_image(image):
    return (image / 127.5) - 1.0

def augment_image(image):
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
    if random.choice([True, False]):
        angle = random.randint(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.choice([True, False]):
        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

def normalize_map(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def extract_structural_maps(image, filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast enhancement before structural extraction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # --- Canny Edges ---
    edges = cv2.Canny(gray, 100, 200)
    edges_norm = normalize_map(edges)

    # --- Harris Corners ---
    gray_float = np.float32(gray)
    harris = cv2.cornerHarris(gray_float, blockSize=3, ksize=5, k=0.04)
    harris = cv2.dilate(harris, None)
    harris_norm = normalize_map(harris)

    # --- ORB Keypoints ---
    orb = cv2.ORB_create(nfeatures=500)
    kp = orb.detect(gray, None)
    keypoint_map = np.zeros_like(gray, dtype=np.float32)
    orb_overlay = image.copy()
    for point in kp:
        x, y = int(point.pt[0]), int(point.pt[1])
        if 0 <= y < keypoint_map.shape[0] and 0 <= x < keypoint_map.shape[1]:
            keypoint_map[y, x] = 1.0
            cv2.circle(orb_overlay, (x, y), 1, (0, 255, 0), -1)

    keypoint_map = cv2.GaussianBlur(keypoint_map, (3, 3), 0)
    keypoint_norm = normalize_map(keypoint_map)

    # Save debug maps
    base_name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_edges.png"), (edges_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_harris.png"), (harris_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_orb_map.png"), (keypoint_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_orb_overlay.png"), orb_overlay)

    return edges_norm, harris_norm, keypoint_norm


# Process each damaged variant and match with original
def get_original_from_variant(damaged_filename):
    name, ext = os.path.splitext(damaged_filename)
    parts = name.split("_")
    if parts[-1].isdigit():
        original_base = "_".join(parts[:-1])
    else:
        original_base = name
    return original_base + ext

valid_exts = ['.jpg', '.jpeg', '.png']

def process_dataset():
    for damaged_filename in os.listdir(DAMAGED_FOLDER):
        name, ext = os.path.splitext(damaged_filename)
        if ext.lower() not in valid_exts:
            continue

        # Extract original filename (everything before the first underscore)
        original_filename = get_original_from_variant(damaged_filename)
        original_path = os.path.join(ORIGINAL_FOLDER, original_filename)
        damaged_path = os.path.join(DAMAGED_FOLDER, damaged_filename)

        # If .jpg not found, try .png or .jpeg
        if not os.path.exists(original_path):
            print(f"No matching original for: {damaged_filename}")
            continue

        original_img = cv2.imread(original_path)
        damaged_img = cv2.imread(damaged_path)

        if original_img is None or damaged_img is None:
            print(f"Could not read image pair: {damaged_filename}")
            continue

        try:
            original_img = cv2.resize(original_img, IMAGE_SIZE)
            damaged_img = cv2.resize(damaged_img, IMAGE_SIZE)

            original_img = augment_image(original_img)
            damaged_img = augment_image(damaged_img)

            # Save normalized RGB images
            norm_original = normalize_image(original_img)
            norm_damaged = normalize_image(damaged_img)

            out_original = ((norm_original + 1) * 127.5).astype(np.uint8)
            out_damaged = ((norm_damaged + 1) * 127.5).astype(np.uint8)

            cv2.imwrite(os.path.join(PROCESSED_ORIGINAL_FOLDER, damaged_filename), out_original)
            cv2.imwrite(os.path.join(PROCESSED_DAMAGED_FOLDER, damaged_filename), out_damaged)

            # Extract structural maps from damaged image
            edges, harris, keypoints = extract_structural_maps(damaged_img, damaged_filename)

            # Prepare 6-channel tensor
            damaged_float = damaged_img.astype(np.float32) / 255.0
            damaged_rgb = cv2.cvtColor(damaged_float, cv2.COLOR_BGR2RGB)
            damaged_rgb = np.transpose(damaged_rgb, (2, 0, 1))  # (3, H, W)

            structural_tensor = np.concatenate([
                damaged_rgb,
                edges[np.newaxis, :, :],
                harris[np.newaxis, :, :],
                keypoints[np.newaxis, :, :]
            ], axis=0)

            # Save .npy
            output_npy = os.path.join(STRUCTURE_MAP_FOLDER, damaged_filename.replace(ext, ".npy"))
            np.save(output_npy, structural_tensor.astype(np.float32))

        except Exception as e:
            print(f"Error processing {damaged_filename}: {e}")

if __name__ == "__main__":
    process_dataset()
    print("Processed all images successfully")
