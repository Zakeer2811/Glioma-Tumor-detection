import os
import glob
import shutil
import random
import yaml
from ultralytics import YOLO
import torch
import requests

# ======================
# 1. SETUP: Copy dataset to working directory
# ======================
INPUT_DIR = "/kaggle/input/mini-project2"  # read-only input
WORKING_DIR = "/kaggle/working/mini_project2"

if not os.path.exists(WORKING_DIR):
    print(f"Copying dataset from {INPUT_DIR} to {WORKING_DIR} ...")
    shutil.copytree(INPUT_DIR, WORKING_DIR)
    print("Copy complete.")
else:
    print("Dataset already exists in working directory.")

# ======================
# 2. SPLIT DATASET INTO TRAIN/VAL/TEST
# ======================
# Updated paths: Assuming images and labels are under 'all/'
src_images_dir = os.path.join(WORKING_DIR, "all", "images")
src_labels_dir = os.path.join(WORKING_DIR, "all", "labels")

# Destination directories for train/val/test splits:
split_dirs = {
    "train": {"images": os.path.join(WORKING_DIR, "train", "images"),
              "labels": os.path.join(WORKING_DIR, "train", "labels")},
    "val":   {"images": os.path.join(WORKING_DIR, "val", "images"),
              "labels": os.path.join(WORKING_DIR, "val", "labels")},
    "test":  {"images": os.path.join(WORKING_DIR, "test", "images"),
              "labels": os.path.join(WORKING_DIR, "test", "labels")},
}

# Create directories if they don't exist
for phase in split_dirs:
    for folder in split_dirs[phase].values():
        os.makedirs(folder, exist_ok=True)

# Get list of all image files (assuming jpg and png)
image_files = sorted(glob.glob(os.path.join(src_images_dir, "*.jpg")) +
                     glob.glob(os.path.join(src_images_dir, "*.png")))
total_images = len(image_files)
print(f"Total images found: {total_images}")

# Set split ratios: 80% train, 10% val, 10% test
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random.shuffle(image_files)
n_train = int(total_images * train_ratio)
n_val = int(total_images * val_ratio)
n_test = total_images - n_train - n_val  # remaining images

train_files = image_files[:n_train]
val_files = image_files[n_train:n_train+n_val]
test_files = image_files[n_train+n_val:]

def copy_files(file_list, phase):
    for img_path in file_list:
        basename = os.path.basename(img_path)
        # Copy image
        shutil.copy(img_path, os.path.join(split_dirs[phase]["images"], basename))
        # Copy corresponding label (if exists)
        label_name = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(src_labels_dir, label_name)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(split_dirs[phase]["labels"], label_name))

print(f"Copying {len(train_files)} images to train ...")
copy_files(train_files, "train")
print(f"Copying {len(val_files)} images to val ...")
copy_files(val_files, "val")
print(f"Copying {len(test_files)} images to test ...")
copy_files(test_files, "test")

# ======================
# 3. DATA CONFIGURATION DICTIONARY
# ======================
data_config = {
    "train": os.path.relpath(split_dirs["train"]["images"], WORKING_DIR),
    "val": os.path.relpath(split_dirs["val"]["images"], WORKING_DIR),
    "test": os.path.relpath(split_dirs["test"]["images"], WORKING_DIR),
    "nc": 1,  # Update if you have more classes
    "names": ["Tumor"]  # Change class names as needed
}
print("Data configuration:")
print(data_config)

# (Optional) Save the data configuration as YAML
yaml_path = os.path.join(WORKING_DIR, "data_config.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)
print(f"Data configuration saved to {yaml_path}")

# ======================
# 4. TRAIN YOLO11 MODEL WITH OPTIMIZED PARAMETERS
# ======================
# Choose one of: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
# Example using 'yolo11m.pt'
MODEL_VARIANT = "m"
models_dir = os.path.join(WORKING_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
MODEL_WEIGHTS_PATH = os.path.join(models_dir, f"yolo11{MODEL_VARIANT}.pt")


# If weights don't exist, download them (update the URL to the correct file location)
if not os.path.exists(MODEL_WEIGHTS_PATH):
    weight_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"  # Update this URL accordingly
    print(f"Downloading {weight_url} to '{MODEL_WEIGHTS_PATH}' ...")
    r = requests.get(weight_url, allow_redirects=True)
    if r.status_code == 200:
        with open(MODEL_WEIGHTS_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    else:
        raise FileNotFoundError(f"Unable to download model weights from {weight_url}")

# Set hyperparameters suited for a T4 GPU
BATCH_SIZE = 8
IMAGE_SIZE = 512

TRAIN_CONFIG = {
    "epochs": 300,
    "imgsz": IMAGE_SIZE,
    "batch": BATCH_SIZE,
    "optimizer": "AdamW",
    "lr0": 0.0003,
    "lrf": 0.01,
    "patience": 20,
    "mosaic": 0.8,
    "mixup": 0.2,
    "degrees": 20.0,
    "shear": 2.5,
    "dropout": 0.1,
    "cache": False,
}

OUTPUT_PROJECT = os.path.join(WORKING_DIR, "results")
RUN_NAME = "yolo11m_train_run"

# Clean previous run folder if exists
output_run_dir = os.path.join(OUTPUT_PROJECT, RUN_NAME)
if os.path.exists(output_run_dir):
    shutil.rmtree(output_run_dir)
    print(f"Removed existing output folder: {output_run_dir}")

# Initialize model using the weights file path
model = YOLO(MODEL_WEIGHTS_PATH)

train_args = {
    "data": yaml_path,
    "epochs": TRAIN_CONFIG["epochs"],
    "imgsz": TRAIN_CONFIG["imgsz"],
    "batch": TRAIN_CONFIG["batch"],
    "device": "0",
    "optimizer": TRAIN_CONFIG["optimizer"],
    "lr0": TRAIN_CONFIG["lr0"],
    "lrf": TRAIN_CONFIG["lrf"],
    "patience": TRAIN_CONFIG["patience"],
    "mosaic": TRAIN_CONFIG["mosaic"],
    "mixup": TRAIN_CONFIG["mixup"],
    "degrees": TRAIN_CONFIG["degrees"],
    "shear": TRAIN_CONFIG["shear"],
    "fliplr": 0.5,
    "flipud": 0.1,
    "cos_lr": True,
    "weight_decay": 0.0005,
    "dropout": TRAIN_CONFIG["dropout"],
    "cache": TRAIN_CONFIG["cache"],
    "project": OUTPUT_PROJECT,
    "name": RUN_NAME,
}


print("Starting training...")
results = model.train(**train_args)

# Validate the best checkpoint
best_ckpt = model.trainer.best  # path to best model checkpoint
print(f"Best model checkpoint: {best_ckpt}")
best_model = YOLO(best_ckpt)
metrics = best_model.val()
print(f"\nFinal mAP50: {metrics.box.map50:.3f}")
if metrics.box.map50 >= 0.97:
    print(" Target mAP achieved!")
else:
    print(" mAP did not reach 0.97. Consider further tuning.")

# ======================
# 5. EXPORT BEST MODEL TO ONNX FORMAT
# ======================
onnx_path = os.path.join(OUTPUT_PROJECT, RUN_NAME, "best_model.onnx")
print(f"Exporting best model to ONNX: {onnx_path}")
best_model.export(format="onnx", opset=12, dynamic=True, simplify=True, file=onnx_path)
print("ONNX export complete.")

# ======================
# 6. SAVE TRAINING RESULTS
# ======================
results_path = os.path.join(OUTPUT_PROJECT, RUN_NAME, "training_results.yaml")
with open(results_path, "w") as f:
    yaml.dump(results, f)
print(f"Training results saved to {results_path}")
