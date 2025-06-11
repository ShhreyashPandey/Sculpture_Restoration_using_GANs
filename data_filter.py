import os

folder_path = "images"  # replace if needed

# Extensions to KEEP (only .jpg)
keep_ext = {".jpg"}

deleted_files = 0

for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)

    if not os.path.isfile(full_path):
        continue

    ext = os.path.splitext(filename)[1].lower()

    # Delete if not in keep list
    if ext not in keep_ext:
        os.remove(full_path)
        deleted_files += 1
        print(f"Deleted: {filename}")

print(f"\nDone. {deleted_files} non-.jpg files removed from '{folder_path}'.")
