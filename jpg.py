import os

image_dir = "images"  # or the full path if needed

for filename in os.listdir(image_dir):
    full_path = os.path.join(image_dir, filename)

    # Only rename files that end with .jpg.jpg
    if filename.endswith(".jpg.jpg"):
        new_name = filename.replace(".jpg.jpg", ".jpg")
        new_path = os.path.join(image_dir, new_name)
        os.rename(full_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
