import os
from collections import defaultdict

# Set your folder path here
folder_path = "images"  

# Dictionary to count extensions
ext_counts = defaultdict(int)

for filename in os.listdir(folder_path):
    if not os.path.isfile(os.path.join(folder_path, filename)):
        continue

    # Get extension (including dot), or 'no_extension'
    ext = os.path.splitext(filename)[1].lower()
    if ext == "":
        ext = "no_extension"
    
    ext_counts[ext] += 1

# Print grouped result
print("\nFile Extension Summary:\n")
for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
    print(f"{ext:>15}: {count} file(s)")
