import os
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from training import Generator, STRUCTURE_FOLDER, IMAGE_SIZE, device  # reusing constants

# Create output directory
os.makedirs("output", exist_ok=True)

# Load trained generator
generator = Generator().to(device)
generator.load_state_dict(torch.load("checkpoints/best_generator.pth", map_location=device))
generator.eval()

# Pick one damaged input from structure maps
test_file = os.listdir(STRUCTURE_FOLDER)[4620]
structure_path = os.path.join(STRUCTURE_FOLDER, test_file)

# Load 6-channel input
input_tensor = np.load(structure_path)
input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, 6, H, W)

# Generate output
with torch.no_grad():
    output = generator(input_tensor)
    output = (output + 1) / 2  # from [-1, 1] to [0, 1]
    save_image(output, f"output/generated.png")

print(f"Output saved")
