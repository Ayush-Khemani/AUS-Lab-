"""
Camera Sphere Detection → CSV Export

Processes camera images and exports sphere centers in CSV format
that matches your teammate's LiDAR format.

Output format (exactly what your teammate needs):
    fn,cx,cy,cz
    test_fn47,0.032,0.040,0.610
    test_fn48,0.031,0.041,0.612
    ...

The frame names (fn) are extracted from your image filenames
and matched with your teammate's LiDAR point clouds.

Usage:
    python camera_detection_to_csv.py

Output:
    outputs/centers_cam.csv
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

# Your camera intrinsic parameters
CAMERA_MATRIX = np.array([
    [625, 0, 480],
    [0, 625, 300],
    [0, 0, 1]
], dtype=np.float32)

SPHERE_RADIUS = 0.25  # meters

# Folders
IMAGE_DIR = "Data/Img"              # Where your images are (CORRECT PATH)
OUTPUT_DIR = "outputs"               # Where to save CSV
OUTPUT_CSV = f"{OUTPUT_DIR}/centers_cam.csv"

# Camera detection parameters
OPTIMIZE = True                      # Refine sphere center with optimization
CANNY_LOW = 50                       # Edge detection lower threshold
CANNY_HIGH = 150                     # Edge detection upper threshold

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_frame_number(filename):
    """
    Extract frame number from image filename.
    
    Examples:
        Dev2_Image_w960_h600_fn119.jpg → fn119
        test_fn47.jpg → fn47
        frame_001.jpg → fn001
    
    Returns: string like "fn119" or None if extraction fails
    """
    # Try to find fn### pattern (works for your format)
    match = re.search(r'fn(\d+)', filename)
    if match:
        return f"fn{match.group(1)}"
    
    # Try to find any number sequence of length 2+
    match = re.search(r'(\d{2,})', filename)
    if match:
        num = match.group(1)
        # Remove leading zeros for cleaner naming, but keep at least 2 digits
        return f"fn{num.lstrip('0') or '0'}"
    
    return None


def get_image_files(image_dir):
    """Get all image files from directory."""
    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG')
    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    return sorted(images)


def process_images_to_csv(image_dir, output_csv, camera_matrix, sphere_radius):
    """
    Process all images and save sphere centers to CSV.
    
    Returns: number of successfully processed images
    """
    
    # Create output directory
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    # Initialize camera module
    cam = CameraCalibration(camera_matrix, sphere_radius)
    
    # Get images
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"ERROR: No images found in {image_dir}")
        print("Expected image files like: Dev2_Image_w960_h600_fn119.jpg")
        return 0
    
    print(f"\n{'='*70}")
    print(f"CAMERA SPHERE DETECTION → CSV EXPORT")
    print(f"{'='*70}")
    print(f"\nFound {len(image_files)} image files\n")
    
    # CSV header
    rows = [["fn", "cx", "cy", "cz"]]
    
    successful = 0
    failed = 0
    
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, filename)
        
        # Extract frame number
        frame_num = extract_frame_number(filename)
        if frame_num is None:
            print(f"[{idx}/{len(image_files)}] {filename}")
            print(f"  ✗ Could not extract frame number from filename")
            failed += 1
            continue
        
        print(f"[{idx}/{len(image_files)}] {filename}", end="")
        print(f" (frame: {frame_num})", end="")
        
        try:
            # Process image
            result = cam.process_image(image_path, optimize=OPTIMIZE)
            
            if result is None:
                print(" ✗ Detection failed\n")
                failed += 1
                continue
            
            # Extract sphere center
            sphere_center = result['sphere_center']
            cx, cy, cz = sphere_center[0], sphere_center[1], sphere_center[2]
            
            # Add to CSV rows
            rows.append([frame_num, f"{cx:.6f}", f"{cy:.6f}", f"{cz:.6f}"])
            
            print(f" ✓")
            print(f"     Center: [{cx:.3f}, {cy:.3f}, {cz:.3f}]")
            print(f"     Contour points: {len(result['contour_points'])}")
            
            successful += 1
            
        except Exception as e:
            print(f" ✗ ERROR: {e}\n")
            failed += 1
    
    # Write CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Successful: {successful} images")
    print(f"Failed:     {failed} images")
    print(f"Total:      {len(image_files)} images\n")
    print(f"✓ CSV saved to: {output_csv}")
    print(f"\nCSV Format (ready for your teammate):")
    print(f"{'─'*70}")
    
    # Show first few rows
    if len(rows) > 1:
        print(f"{rows[0][0]},{rows[0][1]},{rows[0][2]},{rows[0][3]}")
        for row in rows[1:min(6, len(rows))]:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]}")
        if len(rows) > 6:
            print(f"... ({len(rows)-1} total frames)")
    
    print(f"{'─'*70}\n")
    
    return successful


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "CAMERA SPHERE DETECTION → CSV EXPORT" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check configuration
    print(f"\n📋 Configuration:")
    print(f"  Camera matrix K: [[625, 0, 480], [0, 625, 300], [0, 0, 1]]")
    print(f"  Sphere radius: {SPHERE_RADIUS} m")
    print(f"  Image folder: {IMAGE_DIR}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Optimization: {OPTIMIZE}")
    
    # Process images
    try:
        success_count = process_images_to_csv(
            IMAGE_DIR,
            OUTPUT_CSV,
            CAMERA_MATRIX,
            SPHERE_RADIUS
        )
        
        if success_count > 0:
            print("✓ COMPLETE!\n")
            print("📤 Next steps:")
            print(f"  1. Send {OUTPUT_CSV} to your teammate")
            print(f"  2. They will use it in compute_extrinsic.py")
            print(f"  3. Make sure frame names match their centers_valid.csv\n")
            return 0
        else:
            print("✗ No images were processed successfully\n")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())