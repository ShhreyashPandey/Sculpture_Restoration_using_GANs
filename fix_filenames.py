import os

image_dir = "images"

for filename in os.listdir(image_dir):
    full_path = os.path.join(image_dir, filename)

    # Skip if it's not a file
    if not os.path.isfile(full_path):
        continue

    # Fix double .jpg.jpg
    if filename.endswith(".jpg.jpg"):
        new_name = filename.replace(".jpg.jpg", ".jpg")
    # Add .jpg if no extension
    elif '.' not in filename:
        new_name = filename + ".jpg"
    else:
        continue  # No rename needed

    new_path = os.path.join(image_dir, new_name)
    os.rename(full_path, new_path)
    print(f"Renamed: {filename} -> {new_name}")
