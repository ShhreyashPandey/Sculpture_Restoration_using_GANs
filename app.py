from flask import Flask, request, render_template, send_from_directory, url_for
import os
import torch
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from postprocess import apply_inpainting, upscale_with_srgan
from training import Generator
from preprocess import extract_structural_maps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

print("Static directories created:")
print(f"- Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
print(f"- Output folder: {os.path.abspath(app.config['OUTPUT_FOLDER'])}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load generator
generator = Generator().to(device)
generator.load_state_dict(torch.load("checkpoints/best_generator.pth", map_location=device))
generator.eval()
print("Model loaded successfully")

def prepare_input(image_path):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    print(f"Image loaded successfully, shape: {img.shape}")
    
    img = cv2.resize(img, (256, 256))
    edges, harris, keypoints = extract_structural_maps(img, os.path.basename(image_path))
    
    img_rgb = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))  # (3, H, W)

    structure_tensor = np.concatenate([
        img_rgb,
        edges[np.newaxis, :, :],
        harris[np.newaxis, :, :],
        keypoints[np.newaxis, :, :]
    ], axis=0)

    input_tensor = torch.tensor(structure_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    return input_tensor

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("No file part in request")
            return render_template('index.html', error="No file uploaded")
        
        img_file = request.files['image']
        if img_file.filename == '':
            print("No selected file")
            return render_template('index.html', error="No file selected")

        if img_file:
            try:
                filename = secure_filename(img_file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img_file.save(upload_path)
                print(f"Image saved to: {upload_path}")

                # Inference pipeline
                input_tensor = prepare_input(upload_path)
                print("Input tensor prepared successfully")
                
                with torch.no_grad():
                    output = generator(input_tensor)
                    output = (output[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
                    output = np.clip(output, 0, 255).astype(np.uint8)
                print("Generator inference completed")

                # Step 1: Inpainting
                inpainted = apply_inpainting(output)
                print("Inpainting completed")

                # Step 2: SRGAN
                final_output = upscale_with_srgan(inpainted, device)
                print("SRGAN upscaling completed")

                # Save outputs
                restored_filename = f"restored_{filename}"
                restored_path = os.path.join(app.config['OUTPUT_FOLDER'], restored_filename)
                cv2.imwrite(restored_path, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
                print(f"Restored image saved to: {restored_path}")

                # Return paths relative to static folder for template
                return render_template('index.html',
                                    original=f"uploads/{filename}",
                                    restored=f"outputs/{restored_filename}")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return render_template('index.html', error=f"Error processing image: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(host='0.0.0.0', port=5051, debug=True)



