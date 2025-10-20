"""
Data Preparation Script for Skull Detection YOLOv8 Training
This script helps organize images for labeling and training
"""

import os
import shutil
import random
from pathlib import Path

def prepare_dataset(source_images_dir, dataset_dir, train_ratio=0.8):
    """
    Organize images into train/val splits for YOLOv8 training
    
    Args:
        source_images_dir: Directory containing all your skull images
        dataset_dir: Directory where the organized dataset will be created
        train_ratio: Ratio of images to use for training (0.8 = 80% train, 20% val)
    """
    
    # Create paths
    source_path = Path(source_images_dir)
    dataset_path = Path(dataset_dir)
    
    train_images_dir = dataset_path / "images" / "train"
    val_images_dir = dataset_path / "images" / "val"
    train_labels_dir = dataset_path / "labels" / "train"
    val_labels_dir = dataset_path / "labels" / "val"
    
    # Create directories if they don't exist
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {source_images_dir}")
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Copy images to respective directories
    for img_file in train_images:
        shutil.copy2(img_file, train_images_dir / img_file.name)
    
    for img_file in val_images:
        shutil.copy2(img_file, val_images_dir / img_file.name)
    
    print("Dataset preparation complete!")
    print(f"Next step: Use LabelImg to annotate images in:")
    print(f"  - {train_images_dir}")
    print(f"  - {val_images_dir}")
    print(f"Save labels to:")
    print(f"  - {train_labels_dir}")
    print(f"  - {val_labels_dir}")

def install_labelimg():
    """
    Instructions for installing LabelImg
    """
    print("\n=== LabelImg Installation Instructions ===")
    print("1. Install LabelImg using pip:")
    print("   pip install labelImg")
    print("\n2. Or download from GitHub:")
    print("   https://github.com/heartexlabs/labelImg")
    print("\n3. Run LabelImg:")
    print("   labelImg")
    print("\n4. In LabelImg:")
    print("   - Open Dir: Select your images folder")
    print("   - Change Save Dir: Select your labels folder")
    print("   - Choose YOLO format")
    print("   - Draw bounding boxes around skulls")
    print("   - Label them as 'skull'")
    print("   - Save each annotation")

if __name__ == "__main__":
    # Example usage
    print("=== Skull Detection Dataset Preparation ===")
    
    # You need to specify where your skull images are located
    source_images = input("Enter path to your skull images directory: ").strip()
    
    if not source_images:
        print("Using default example path...")
        source_images = "C:/Users/ngoch/Downloads/skull_images"  # Update this path
    
    dataset_directory = "skull_detection_dataset"
    
    if os.path.exists(source_images):
        prepare_dataset(source_images, dataset_directory)
    else:
        print(f"Directory {source_images} not found!")
        print("Please create a directory with your skull images first.")
    
    install_labelimg()
