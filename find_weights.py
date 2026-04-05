"""
Find trained YOLO model weights
"""

import os
import glob

# Check if training directory exists
if os.path.exists('runs/cs_detection'):
    print("Available runs:")
    runs = glob.glob('runs/cs_detection/*')
    for run in runs:
        print(f"  {run}")
        weights = glob.glob(f"{run}/weights/*.pt")
        if weights:
            print(f"    Weights found:")
            for w in weights:
                print(f"      - {w}")
        else:
            print(f"    No weights found")
else:
    print("No training runs found!")
    print("\nSearching for any .pt files...")
    all_pt = glob.glob('**/*.pt', recursive=True)
    for pt in all_pt:
        print(f"  {pt}")
