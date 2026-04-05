"""
Dataset Split Script for YOLO Training
Splits images and labels into train/val sets (80/20)
"""

import os
import shutil
from pathlib import Path
import random

# Configuration
DATASET_ROOT = "817_pics"  # Change this if your folder name is different
TRAIN_SPLIT = 0.8  # 80% train, 20% val
RANDOM_SEED = 42  # For reproducible splits

def split_dataset(dataset_root, train_ratio=0.8, seed=42):
    """
    Split dataset into train/val folders
    """
    random.seed(seed)

    dataset_path = Path(dataset_root)
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"

    # Verify paths exist
    if not images_path.exists():
        print(f"ERROR: {images_path} not found!")
        return
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found!")
        return

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f'*{ext}')))
        image_files.extend(list(images_path.glob(f'*{ext.upper()}')))

    if not image_files:
        print("ERROR: No images found!")
        return

    print(f"Found {len(image_files)} images")

    # Verify corresponding label files exist
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            print(f"WARNING: No label file for {img_file.name}")

    print(f"Valid image-label pairs: {len(valid_pairs)}")

    if not valid_pairs:
        print("ERROR: No valid image-label pairs found!")
        return

    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")

    # Create directory structure
    train_images_dir = dataset_path / "images" / "train"
    train_labels_dir = dataset_path / "labels" / "train"
    val_images_dir = dataset_path / "images" / "val"
    val_labels_dir = dataset_path / "labels" / "val"

    # Create directories
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

    # Copy files to train folder
    print("\nCopying training files...")
    for img_file, label_file in train_pairs:
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(label_file, train_labels_dir / label_file.name)

    # Copy files to val folder
    print("Copying validation files...")
    for img_file, label_file in val_pairs:
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(label_file, val_labels_dir / label_file.name)

    print("\n✓ Dataset split completed!")
    print(f"\nNew structure:")
    print(f"  {dataset_root}/")
    print(f"    images/")
    print(f"      train/  ({len(train_pairs)} images)")
    print(f"      val/    ({len(val_pairs)} images)")
    print(f"    labels/")
    print(f"      train/  ({len(train_pairs)} labels)")
    print(f"      val/    ({len(val_pairs)} labels)")

    # Create data.yaml
    yaml_content = f"""# CS Enemy Detection Dataset
path: {dataset_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: enemy

# Number of classes
nc: 1
"""

    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Created data.yaml at: {yaml_path}")
    print("\nYou can now use this dataset for YOLO training!")
    print(f"data.yaml path: {yaml_path.absolute()}")

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Dataset Split Script")
    print("=" * 60)
    print()

    # Check if dataset folder exists
    if not os.path.exists(DATASET_ROOT):
        print(f"ERROR: Dataset folder '{DATASET_ROOT}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease:")
        print("1. Make sure you're in the correct directory")
        print("2. Or update DATASET_ROOT variable in the script")
        exit(1)

    # Run split
    split_dataset(DATASET_ROOT, TRAIN_SPLIT, RANDOM_SEED)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
