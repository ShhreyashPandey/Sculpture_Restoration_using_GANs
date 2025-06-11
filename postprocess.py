import os
import cv2
import numpy as np
from PIL import Image
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# === CONFIG ===
INPUT_IMAGE_PATH = "output/generated.png"
INPAINTED_IMAGE_PATH = "output/inpainted.png"
FINAL_OUTPUT_PATH = "output/final_restored.png"
SRGAN_WEIGHTS_PATH = "weights/RealESRGAN_x4.pth"

# === STEP 1: NAVIER-STOKES INPAINTING ===
def apply_inpainting(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return inpainted

def upscale_with_srgan(image_np: np.ndarray, device: torch.device) -> np.ndarray:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=SRGAN_WEIGHTS_PATH,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    img = Image.fromarray(image_np)
    output, _ = upsampler.enhance(np.array(img), outscale=4)
    return output


# === MAIN POSTPROCESSING PIPELINE ===
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load generated Pix2Pix output
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE_PATH}")

    gen_img = cv2.imread(INPUT_IMAGE_PATH)
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)

    # Step 1: Inpainting
    print("Performing inpainting...")
    inpainted_img = apply_inpainting(gen_img)
    cv2.imwrite(INPAINTED_IMAGE_PATH, cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR))
    print(f"Inpainted image saved to {INPAINTED_IMAGE_PATH}")

    # Step 2: SRGAN Upscaling
    print("Applying SRGAN Super-Resolution...")
    final_img = upscale_with_srgan(inpainted_img, device)
    cv2.imwrite(FINAL_OUTPUT_PATH, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"Final enhanced image saved to {FINAL_OUTPUT_PATH}")
