import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_msssim import ssim

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATOR_WEIGHTS = "checkpoints/best_generator.pth"
STRUCTURE_FOLDER = "preprocessed/structure_maps"
ORIGINAL_FOLDER = "preprocessed/original"
NUM_TESTS = 50

# === Load Generator ===
from training import Generator
generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(GENERATOR_WEIGHTS, map_location=DEVICE))
generator.eval()

# === Metrics ===
fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
fid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous())
])


def edge_preservation_score(real, restored):
    edges_real = cv2.Canny(real, 100, 200)
    edges_fake = cv2.Canny(restored, 100, 200)
    return 1 - np.mean(np.abs(edges_real - edges_fake) / 255)

# === Run Evaluation ===
test_files = os.listdir(ORIGINAL_FOLDER)[:NUM_TESTS]
edge_scores, ssim_scores = [], []

plt.figure(figsize=(12, 8))
for idx, file in enumerate(test_files):
    base = os.path.splitext(file)[0]
    original_path = os.path.join(ORIGINAL_FOLDER, file)
    structure_path = os.path.join(STRUCTURE_FOLDER, base + ".npy")

    # Load data
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    input_tensor = torch.tensor(np.load(structure_path), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = generator(input_tensor)
        out_img = (output[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    # Inpainting (Navier-Stokes)
    gray = cv2.cvtColor(out_img, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)[1]
    inpainted = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_NS)

    # Convert images to tensor for FID & SSIM
    real_tensor = fid_transform(original_rgb).unsqueeze(0).to(DEVICE)
    fake_tensor = fid_transform(inpainted).unsqueeze(0).to(DEVICE)

    fid.update(real_tensor, real=True)
    fid.update(fake_tensor, real=False)

    # SSIM (with resizing to match)
    orig_resized = cv2.resize(original_rgb, (256, 256))
    inpainted_resized = cv2.resize(inpainted, (256, 256))
    orig_tensor = torch.tensor(orig_resized).permute(2, 0, 1).unsqueeze(0).float() / 255
    fake_tensor = torch.tensor(inpainted_resized).permute(2, 0, 1).unsqueeze(0).float() / 255
    ssim_score = ssim(orig_tensor.to(DEVICE), fake_tensor.to(DEVICE), data_range=1.0).item()

    # Edge score
    edge_score = edge_preservation_score(orig_resized, inpainted_resized)

    edge_scores.append(edge_score)
    ssim_scores.append(ssim_score)

    # Visualize
    plt.subplot(NUM_TESTS, 3, idx*3 + 1); plt.imshow(input_tensor[0, :3].cpu().permute(1, 2, 0).numpy()*0.5+0.5); plt.title("Damaged"); plt.axis("off")
    plt.subplot(NUM_TESTS, 3, idx*3 + 2); plt.imshow(inpainted); plt.title("Reconstructed"); plt.axis("off")
    plt.subplot(NUM_TESTS, 3, idx*3 + 3); plt.imshow(orig_resized); plt.title("Original"); plt.axis("off")

plt.tight_layout()
plt.show()

# === Final Evaluation Summary ===
fid_score = fid.compute().item()
print("\nFinal Evaluation on", NUM_TESTS, "Images:")
print(f"Average Edge Preservation Score: {np.mean(edge_scores):.4f}")
print(f"Average SSIM Score:                {np.mean(ssim_scores):.4f}")
print(f"FID Score:                         {fid_score:.4f} (lower is better)")
