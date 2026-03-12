import os
import random
import shutil

# Set your base folder paths
base_path = "C:/Users/1003380/Mine/MLOps/catvdog/raw"  # Change this to your dataset path
image_path = os.path.join(base_path, "images")
label_path = os.path.join(base_path, "labels")

new_path = "C:/Users/1003380/Mine/MLOps/catvdog"  # Base path for the new structure

# Output folders
output_dirs = {
    "train_images": os.path.join(new_path, "train/images"),
    "train_labels": os.path.join(new_path, "train/labels"),
    "val_images": os.path.join(new_path, "val/images"),
    "val_labels": os.path.join(new_path, "val/labels"),
    "test_images": os.path.join(new_path, "test/images"),
    "test_labels": os.path.join(new_path, "test/labels"),
}

# Create output folders if they don't exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# === Split ratios ===
train_ratio = 0.6
val_ratio = 0.4
test_ratio = 0

# Get image files
image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)

# Calculate split counts
total = len(image_files)
train_count = int(total * train_ratio)
val_count = int(total * val_ratio)
test_count = total - train_count - val_count  # handle rounding

# Split data
train_images = image_files[:train_count]
val_images = image_files[train_count:train_count + val_count]
test_images = image_files[train_count + val_count:]

def move_files(file_list, subset):
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_img = os.path.join(image_path, img_file)
        src_lbl = os.path.join(label_path, label_file)

        dst_img = os.path.join(new_path, f"{subset}/images", img_file)
        dst_lbl = os.path.join(new_path, f"{subset}/labels", label_file)

        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"⚠️ Skipping: {img_file} or {label_file} missing.")

# Move files to respective folders
move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")

print("✅ Dataset split complete.")
print(f"Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
