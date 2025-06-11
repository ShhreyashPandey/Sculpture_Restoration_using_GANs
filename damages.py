import os
import cv2
import numpy as np
import random
import itertools
import torch
import torch.nn.functional as F

# Input and Output Directories
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "damaged_images"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert a cv2 image (OpenCV format) to a PyTorch tensor
def cv2_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
    tensor = torch.from_numpy(image).float() / 255.0  # Normalize
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (B, C, H, W)
    return tensor

# Convert a PyTorch tensor back to a cv2 image
def tensor_to_cv2(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    image = (tensor * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
    return image

# add random Gaussian noise
def add_gaussian_noise(image_tensor):
    stddev = random.uniform(0.04, 0.2)# Random noise level
    noise = torch.randn_like(image_tensor).to(device) * stddev
    return (image_tensor + noise).clamp(0, 1)

# blur effect function 
def add_blur(image_tensor):
    kernel_size = random.choice([3, 5, 7])
    pad = kernel_size // 2  # Calculate the necessary padding
    kernel = torch.ones((3, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
    image_blurred = F.conv2d(image_tensor, kernel, padding=pad, groups=3)  # Applying  convolution separately on each channel
    return image_blurred.clamp(0, 1)

#add random scratches 
def add_scratches(image_tensor):
    image_np = tensor_to_cv2(image_tensor)
    h, w, _ = image_np.shape
    for _ in range(random.randint(5, 15)):
        x1, y1 = get_central_coordinates(w, h)
        x2, y2 = get_central_coordinates(w, h)
        thickness = random.randint(1, 3)
        cv2.line(image_np, (x1, y1), (x2, y2), (255, 255, 255), thickness)
    return cv2_to_tensor(image_np)

# add missing patches
def add_missing_patches(image_tensor):
    image_np = tensor_to_cv2(image_tensor)# Convert tensor to image
    h, w, _ = image_np.shape
    for _ in range(random.randint(3, 7)):
        x, y = get_central_coordinates(w - 50, h - 50)
        patch_size = random.randint(20, 100)
        image_np[y:y+patch_size, x:x+patch_size] = (0, 0, 0)
    return cv2_to_tensor(image_np)

def erode_edges(image_tensor):
    image_np = tensor_to_cv2(image_tensor)
    kernel = np.ones((3, 3), np.uint8)
    iterations = random.randint(1, 3)
    eroded = cv2.erode(image_np, kernel, iterations=iterations)
    return cv2_to_tensor(eroded)

def get_central_coordinates(w, h, margin_ratio=0.25):
    x_min = int(w * margin_ratio)
    x_max = int(w * (1 - margin_ratio))
    y_min = int(h * margin_ratio)
    y_max = int(h * (1 - margin_ratio))
    return random.randint(x_min, x_max), random.randint(y_min, y_max)

# Damage function list
damage_functions = [
    ("noise", add_gaussian_noise),
    ("blur", add_blur),
    ("scratches", add_scratches),
    ("patches", add_missing_patches),
    ("erosion", erode_edges),
]

# Processing loop
for filename in os.listdir(INPUT_FOLDER):
    input_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(input_path)

    if image is None:
        print(f"Could not read {filename}")
        continue

    image_tensor = cv2_to_tensor(image)
    name_base = os.path.splitext(filename)[0]
    count = 1

    for r in range(1, 6):
        for combo in itertools.combinations(damage_functions, r):
            img_copy = image_tensor.clone()
            for _, func in combo:
                img_copy = func(img_copy)

            output_image = tensor_to_cv2(img_copy)
            output_path = os.path.join(OUTPUT_FOLDER, f"{name_base}_{count}.jpg")
            cv2.imwrite(output_path, output_image)
            count += 1

print("All images processed with GPU acceleration!")
