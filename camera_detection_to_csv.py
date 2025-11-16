"""
Camera Sphere Detection → CSV Export (FIXED)

NOW ONLY processes images that have matching LiDAR data!

Your LiDAR frames: test_fn47, test_fn48, test_fn61, test_fn65, test_fn76, test_fn82, test_fn84, test_fn118
Your images: Dev0/Dev1/Dev2_Image_w960_h600_fn47.jpg, etc.

This script will:
1. Find LiDAR frame numbers from centers_valid.csv
2. ONLY process images with those frame numbers
3. Export matching CSV for extrinsic calibration
"""

import os
import sys
import csv
import re
import numpy as np
from pathlib import Path

from camera_calibration import CameraCalibration

# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_MATRIX = np.array([
    [625, 0, 480],
    [0, 625, 300],
    [0, 0, 1]
], dtype=np.float32)

SPHERE_RADIUS = 0.25

IMAGE_DIR = "Data/Img"
LIDAR_CSV = "outputs/centers_valid.csv"  # Your teammate's LiDAR results
OUTPUT_DIR = "outputs"
OUTPUT_CSV = f"{OUTPUT_DIR}/centers_cam.csv"

OPTIMIZE = True
CANNY_LOW = 50
CANNY_HIGH = 150

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_frame_number(filename):
    """Extract frame number from image filename."""
    match = re.search(r'fn(\d+)', filename)
    if match:
        return f"fn{match.group(1)}"
    return None


def get_lidar_frame_numbers(lidar_csv_path):
    """
    Read LiDAR CSV and extract the frame numbers.
    Returns: set of frame numbers like {'fn47', 'fn48', 'fn61', ...}
    """
    frame_numbers = set()
    
    if not os.path.exists(lidar_csv_path):
        print(f"ERROR: LiDAR CSV not found: {lidar_csv_path}")
        return frame_numbers
    
    try:
        with open(lidar_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get('fn', '').strip()
                if fn:
                    # Extract just the number part
                    match = re.search(r'fn(\d+)', fn)
                    if match:
                        frame_numbers.add(f"fn{match.group(1)}")
    except Exception as e:
        print(f"ERROR reading LiDAR CSV: {e}")
    
    return frame_numbers


def get_matching_images(image_dir, lidar_frames):
    """
    Get images that have matching LiDAR frame numbers.
    
    Returns: list of (filename, frame_number) tuples
    """
    matching_images = []
    
    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return matching_images
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG')
    
    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(image_extensions):
            continue
        
        frame_num = extract_frame_number(filename)
        if frame_num and frame_num in lidar_frames:
            matching_images.append((filename, frame_num))
    
    return matching_images


def process_images_to_csv():
    """Main processing function."""
    
    print(f"\n{'='*70}")
    print(f"CAMERA SPHERE DETECTION → CSV EXPORT (MATCHING LIDAR FRAMES)")
    print(f"{'='*70}")
    
    # Step 1: Get LiDAR frame numbers
    print(f"\n📋 Step 1: Reading LiDAR frame numbers...")
    lidar_frames = get_lidar_frame_numbers(LIDAR_CSV)
    
    if not lidar_frames:
        print(f"ERROR: Could not read LiDAR frames from {LIDAR_CSV}")
        return False
    
    print(f"  Found {len(lidar_frames)} LiDAR frames: {sorted(lidar_frames)}")
    
    # Step 2: Find matching images
    print(f"\n📋 Step 2: Finding matching images...")
    matching_images = get_matching_images(IMAGE_DIR, lidar_frames)
    
    if not matching_images:
        print(f"ERROR: No matching images found!")
        print(f"  LiDAR frames: {sorted(lidar_frames)}")
        print(f"  Image folder: {IMAGE_DIR}")
        return False
    
    print(f"  Found {len(matching_images)} matching images:")
    for filename, frame_num in matching_images:
        print(f"    - {filename} → {frame_num}")
    
    # Step 3: Process images
    print(f"\n📋 Step 3: Processing images...")
    cam = CameraCalibration(CAMERA_MATRIX, SPHERE_RADIUS)
    
    rows = [["fn", "cx", "cy", "cz"]]
    successful = 0
    failed = 0
    
    for idx, (filename, frame_num) in enumerate(matching_images, 1):
        image_path = os.path.join(IMAGE_DIR, filename)
        
        print(f"  [{idx}/{len(matching_images)}] {filename}", end="")
        
        try:
            result = cam.process_image(image_path, optimize=OPTIMIZE)
            
            if result is None:
                print(" ✗ Detection failed")
                failed += 1
                continue
            
            sphere_center = result['sphere_center']
            cx, cy, cz = sphere_center[0], sphere_center[1], sphere_center[2]
            
            # Convert to teammate's format: add "test_" prefix
            # Your frame: "fn47" → Teammate's frame: "test_fn47"
            test_frame_num = f"test_{frame_num}"
            rows.append([test_frame_num, f"{cx:.6f}", f"{cy:.6f}", f"{cz:.6f}"])
            
            print(f" ✓")
            successful += 1
            
        except Exception as e:
            print(f" ✗ ERROR: {e}")
            failed += 1
    
    # Step 4: Write CSV
    print(f"\n📋 Step 4: Writing CSV...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(matching_images)}")
    print(f"Failed:     {failed}/{len(matching_images)}")
    print(f"\n✓ CSV saved to: {OUTPUT_CSV}")
    print(f"\nCSV Content (first 10 rows):")
    print(f"{'─'*70}")
    
    if len(rows) > 1:
        for row in rows[:min(11, len(rows))]:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]}")
        if len(rows) > 11:
            print(f"... ({len(rows)-1} total frames)")
    
    print(f"{'─'*70}\n")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "CAMERA DETECTION → CSV (MATCHING LIDAR)" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        success = process_images_to_csv()
        
        if success:
            print("✓ COMPLETE!\n")
            print("📤 Next steps:")
            print(f"  1. Verify CSV looks correct")
            print(f"  2. Send to teammate for compute_extrinsic.py")
            print(f"  3. Frame names now match her LiDAR data!\n")
            return 0
        else:
            print("✗ FAILED\n")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())