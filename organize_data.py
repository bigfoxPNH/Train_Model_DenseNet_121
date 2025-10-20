"""
Script để tổ chức dữ liệu cho DenseNet-121 training
Tạo cấu trúc thư mục cropped_out với train và test sets
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Tạo cấu trúc thư mục cho dữ liệu"""
    base_dir = Path("cropped_out")
    
    # Tạo các thư mục chính
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    
    # Tạo các thư mục con cho train
    train_qualified = train_dir / "qualified"
    train_unqualified = train_dir / "unqualified"
    
    # Tạo các thư mục con cho test
    test_qualified = test_dir / "qualified"
    test_unqualified = test_dir / "unqualified"
    
    # Tạo tất cả các thư mục
    for directory in [train_qualified, train_unqualified, test_qualified, test_unqualified]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return {
        'train_qualified': train_qualified,
        'train_unqualified': train_unqualified,
        'test_qualified': test_qualified,
        'test_unqualified': test_unqualified
    }

def copy_training_data(directories):
    """Copy dữ liệu training từ Downloads"""
    source_qualified = Path("Downloads/Qualified_PNG")
    source_unqualified = Path("Downloads/Unqualified_PNG")
    
    if source_qualified.exists():
        print(f"Copying qualified training data from {source_qualified}")
        for img_file in source_qualified.glob("*.png"):
            dest_file = directories['train_qualified'] / img_file.name
            shutil.copy2(img_file, dest_file)
        print(f"Copied {len(list(source_qualified.glob('*.png')))} qualified training images")
    
    if source_unqualified.exists():
        print(f"Copying unqualified training data from {source_unqualified}")
        for img_file in source_unqualified.glob("*.png"):
            dest_file = directories['train_unqualified'] / img_file.name
            shutil.copy2(img_file, dest_file)
        print(f"Copied {len(list(source_unqualified.glob('*.png')))} unqualified training images")

def create_test_data(directories):
    """Tạo test data bằng cách split từ training data"""
    import random
    
    # Set seed để reproducible
    random.seed(42)
    
    # Split qualified data
    qualified_files = list(directories['train_qualified'].glob("*.png"))
    test_size = int(0.2 * len(qualified_files))  # 20% cho test
    test_qualified_files = random.sample(qualified_files, test_size)
    
    for file in test_qualified_files:
        dest_file = directories['test_qualified'] / file.name
        shutil.move(file, dest_file)
    
    print(f"Moved {len(test_qualified_files)} qualified images to test set")
    
    # Split unqualified data
    unqualified_files = list(directories['train_unqualified'].glob("*.png"))
    test_size = int(0.2 * len(unqualified_files))  # 20% cho test
    test_unqualified_files = random.sample(unqualified_files, test_size)
    
    for file in test_unqualified_files:
        dest_file = directories['test_unqualified'] / file.name
        shutil.move(file, dest_file)
    
    print(f"Moved {len(test_unqualified_files)} unqualified images to test set")

def print_data_summary(directories):
    """In thống kê dữ liệu"""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    for name, path in directories.items():
        count = len(list(path.glob("*.png")))
        print(f"{name}: {count} images")
    
    train_total = len(list(directories['train_qualified'].glob("*.png"))) + \
                  len(list(directories['train_unqualified'].glob("*.png")))
    test_total = len(list(directories['test_qualified'].glob("*.png"))) + \
                 len(list(directories['test_unqualified'].glob("*.png")))
    
    print(f"\nTotal training images: {train_total}")
    print(f"Total test images: {test_total}")
    print(f"Total images: {train_total + test_total}")

if __name__ == "__main__":
    print("Organizing data for DenseNet-121 training...")
    
    # Tạo cấu trúc thư mục
    directories = create_directory_structure()
    
    # Copy dữ liệu training
    copy_training_data(directories)
    
    # Tạo test data
    create_test_data(directories)
    
    # In thống kê
    print_data_summary(directories)
    
    print("\nData organization completed!")
