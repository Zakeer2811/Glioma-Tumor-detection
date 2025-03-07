import os
import glob
import random
import shutil
import nibabel as nib
import cv2
import numpy as np
import yaml
import torch
from concurrent.futures import ThreadPoolExecutor

# ------------------------------
# CONFIGURATION PARAMETERS
# ------------------------------

# Input folder that contains multiple subject folders.
INPUT_ROOT = r"C:\Users\Admin\Downloads\BRATS_2021\Training_data"

# Output root folder where preprocessed files will be saved.
# The script will create subdirectories for train, val, and test splits.
OUTPUT_ROOT = r"C:\D_DRIVE\preprocessed2"

# Define split ratios for train, val, and test (should add to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# If set to None, the middle slice is used; otherwise, use the specified index.
SLICE_INDEX = None

# Define the modalities to process. Now includes: t1, t2, t1ce, flair.
MODALITIES = ["t1", "t2", "t1ce", "flair"]

# The segmentation file should have this substring in its filename.
SEG_KEY = "seg.nii.gz"

# Number of threads for concurrent processing (adjust based on your CPU)
MAX_WORKERS = 8

# ------------------------------
# HELPER FUNCTIONS (with GPU acceleration)
# ------------------------------

def load_nifti_slice(nii_path, slice_index=SLICE_INDEX):
    """
    Loads a NIfTI file and returns a 2D slice.
    If slice_index is None, uses the middle slice along the third axis.
    """
    img = nib.load(nii_path)
    data = img.get_fdata()
    if slice_index is None:
        slice_index = data.shape[2] // 2
    return data[:, :, slice_index]

def normalize_to_uint8(img):
    """
    Normalize a 2D image to the 0-255 range and convert to uint8.
    Uses PyTorch on the GPU (if available) for normalization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor(img, dtype=torch.float32, device=device)
    t_min = torch.min(t)
    t_max = torch.max(t)
    if t_max - t_min == 0:
        norm = torch.zeros_like(t)
    else:
        norm = (t - t_min) / (t_max - t_min) * 255.0
    norm = norm.clamp(0, 255).to(torch.uint8)
    return norm.cpu().numpy()

def get_bounding_box(seg_slice):
    """
    Compute the bounding box for nonzero regions in a segmentation slice.
    Uses PyTorch (and GPU if available) for fast computation.
    Returns (x_min, y_min, x_max, y_max) or None if no nonzero pixels are found.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_seg = torch.tensor(seg_slice, device=device)
    indices = torch.nonzero(t_seg > 0)
    if indices.numel() == 0:
        return None
    y_min = torch.min(indices[:, 0]).item()
    x_min = torch.min(indices[:, 1]).item()
    y_max = torch.max(indices[:, 0]).item()
    x_max = torch.max(indices[:, 1]).item()
    return (x_min, y_min, x_max, y_max)

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert absolute bounding box coordinates to YOLO format (normalized).
    Returns a string in the format:
      "<class> <x_center> <y_center> <width> <height>"
    Here we assume a single class (class index 0).
    """
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0

    # Normalize the coordinates (values between 0 and 1)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_width / img_width
    height_norm = box_height / img_height

    return f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def process_subject_folder(subject_folder, temp_all_dir):
    """
    Process one subject folder:
      - Locate the segmentation file (using SEG_KEY).
      - For each modality in MODALITIES:
          - Locate the MRI scan file (using the modality keyword).
          - Load the chosen slice from the MRI volume.
          - Compute the tumor bounding box using the segmentation slice.
          - Save the processed image as PNG.
          - Save a YOLO-format label file:
               * If a tumor is detected, include the bounding box line.
               * Otherwise, create an empty file (to indicate no tumor).
    
    Returns a list of base names (one per modality) that were processed.
    """
    processed_basenames = []
    
    # Find the segmentation file (assumed to contain SEG_KEY)
    seg_files = glob.glob(os.path.join(subject_folder, f"*{SEG_KEY}"))
    if len(seg_files) == 0:
        print(f"Skipping {subject_folder}: Segmentation file not found.")
        return []
    seg_path = seg_files[0]
    
    try:
        # Load the segmentation slice once.
        seg_slice = load_nifti_slice(seg_path)
        bbox = get_bounding_box(seg_slice)
    except Exception as e:
        print(f"Error processing segmentation for {subject_folder}: {e}")
        return []
    
    # Process each modality.
    for modality in MODALITIES:
        modality_files = glob.glob(os.path.join(subject_folder, f"*{modality}*.nii.gz"))
        if len(modality_files) == 0:
            print(f"Modality {modality} not found in {subject_folder}. Skipping modality.")
            continue

        modality_path = modality_files[0]
        # Construct a unique base name: subjectFolder_modality
        base_name = os.path.basename(os.path.normpath(subject_folder)) + f"_{modality}"
        
        try:
            # Load the modality slice.
            mri_slice = load_nifti_slice(modality_path)
            mri_uint8 = normalize_to_uint8(mri_slice)
            mri_rgb = cv2.cvtColor(mri_uint8, cv2.COLOR_GRAY2RGB)
            img_h, img_w = mri_rgb.shape[:2]
            
            # Determine the YOLO label: if bbox exists, convert it; otherwise, leave label empty.
            if bbox is not None:
                yolo_label = convert_bbox_to_yolo(bbox, img_w, img_h)
            else:
                yolo_label = ""  # empty label file indicates no objects

            # Prepare output directories inside the temporary "all" folder.
            images_dir = os.path.join(temp_all_dir, "images")
            labels_dir = os.path.join(temp_all_dir, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Save image and label.
            img_out_path = os.path.join(images_dir, f"{base_name}.png")
            label_out_path = os.path.join(labels_dir, f"{base_name}.txt")
            cv2.imwrite(img_out_path, mri_rgb)
            with open(label_out_path, "w") as f:
                if yolo_label:
                    f.write(yolo_label + "\n")
            print(f"Processed subject modality: {base_name}")
            processed_basenames.append(base_name)
        except Exception as e:
            print(f"Error processing modality {modality} for {subject_folder}: {e}")
            continue

    return processed_basenames

def find_subject_folders(input_root):
    """
    List all subdirectories in the given INPUT_ROOT.
    Each subdirectory is assumed to be a subject folder.
    """
    subject_folders = []
    for entry in os.listdir(input_root):
        subj_path = os.path.join(input_root, entry)
        if os.path.isdir(subj_path):
            subject_folders.append(subj_path)
    print(f"Found {len(subject_folders)} subject folders.")
    return subject_folders

def split_dataset(basenames, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """
    Randomly split the list of basenames into train, validation, and test sets.
    """
    random.shuffle(basenames)
    n = len(basenames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_set = basenames[:n_train]
    val_set = basenames[n_train:n_train+n_val]
    test_set = basenames[n_train+n_val:]
    print(f"Dataset split into {len(train_set)} train, {len(val_set)} val, {len(test_set)} test images.")
    return train_set, val_set, test_set

def move_files_to_split(temp_all_dir, split_names, split_name, output_root):
    """
    Copy image and label files corresponding to subject modality base names in split_names
    from the temporary "all" folder to the corresponding split folder (train/val/test) under OUTPUT_ROOT.
    """
    src_img_dir = os.path.join(temp_all_dir, "images")
    src_label_dir = os.path.join(temp_all_dir, "labels")
    dest_img_dir = os.path.join(output_root, split_name, "images")
    dest_label_dir = os.path.join(output_root, split_name, "labels")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    for name in split_names:
        src_img = os.path.join(src_img_dir, f"{name}.png")
        src_label = os.path.join(src_label_dir, f"{name}.txt")
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(dest_img_dir, f"{name}.png"))
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(dest_label_dir, f"{name}.txt"))
        else:
            open(os.path.join(dest_label_dir, f"{name}.txt"), 'w').close()
    print(f"Moved {len(split_names)} images to '{split_name}' folder.")

def create_yolo_yaml(train_dir, val_dir, yaml_path, class_names=["tumor"]):
    """
    Create a YOLO data configuration YAML file.
    """
    data_config = {
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(val_dir),
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f)
    print(f"Created YOLO YAML config at: {yaml_path}")

# ------------------------------
# MAIN EXECUTION
# ------------------------------

if __name__ == "__main__":
    random.seed(42)

    # Step 1: Find all subject folders inside INPUT_ROOT.
    subject_folders = find_subject_folders(INPUT_ROOT)

    # Create a temporary directory under OUTPUT_ROOT to store all processed images.
    temp_all_dir = os.path.join(OUTPUT_ROOT, "all")
    os.makedirs(os.path.join(temp_all_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(temp_all_dir, "labels"), exist_ok=True)

    # Step 2: Process each subject folder concurrently using a thread pool.
    all_processed_basenames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_subject_folder, folder, temp_all_dir)
                   for folder in subject_folders]
        for future in futures:
            result = future.result()
            if result:
                all_processed_basenames.extend(result)

    print(f"Total processed images: {len(all_processed_basenames)}")

    # Step 3: Split the processed images into train, val, and test sets.
    train_set, val_set, test_set = split_dataset(all_processed_basenames, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Create output directories for train, val, and test splits.
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "labels"), exist_ok=True)

    # Step 4: Move the processed files from the temporary "all" folder into their respective split directories.
    move_files_to_split(temp_all_dir, train_set, "train", OUTPUT_ROOT)
    move_files_to_split(temp_all_dir, val_set, "val", OUTPUT_ROOT)
    move_files_to_split(temp_all_dir, test_set, "test", OUTPUT_ROOT)

    # Step 5: Create a YOLO YAML configuration file (using the train and val folders).
    yolo_yaml_path = os.path.join(OUTPUT_ROOT, "brats_yolo.yaml")
    train_images_dir = os.path.join(OUTPUT_ROOT, "train", "images")
    val_images_dir = os.path.join(OUTPUT_ROOT, "val", "images")
    create_yolo_yaml(train_images_dir, val_images_dir, yolo_yaml_path, class_names=["tumor"])

    print("Preprocessing complete. Your YOLO dataset is ready for training!")
