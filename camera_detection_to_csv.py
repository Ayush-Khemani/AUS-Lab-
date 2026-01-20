import numpy as np
import cv2
from scipy.optimize import least_squares
import os
import sys
import csv
import re
from typing import Optional, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

# 1. OPTIMIZATION FLAG
OPTIMIZE = True 

# 2. Camera Intrinsics (UPDATED based on your inputs)
# fu = 625, fv = 625, u0 = 480, v0 = 300
CAMERA_MATRIX = np.array([
    [625, 0, 480],      
    [0, 625, 300],      
    [0, 0, 1]
], dtype=np.float32)

# Sphere Radius (meters)
SPHERE_RADIUS = 0.25

# 3. Paths
IMAGE_DIR = "Data/Img"
# Expects the file with the 9 rows you provided
LIDAR_CSV = "outputs/centers_valid.csv" 
OUTPUT_DIR = "outputs"
OUTPUT_CSV = f"{OUTPUT_DIR}/centers_cam.csv"

# 4. Filter
# Only process images containing this string (e.g. "Dev2") to avoid duplicates
CAMERA_FILTER = "Dev2" 

# ============================================================================
# CALIBRATION LOGIC
# ============================================================================

class CameraCalibration:
    def __init__(self, K: np.ndarray, sphere_radius: float):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.sphere_radius = sphere_radius
        
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera coordinates."""
        if points.shape[1] != 2:
            raise ValueError("Points must be (N, 2)")
        # Add homogeneous coordinate (u, v, 1)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        # Apply inverse camera matrix
        normalized = (self.K_inv @ points_h.T).T
        return normalized
    
    def detect_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest ellipse contour in the image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Assume the sphere is the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.squeeze()
        
        if len(contour_points.shape) < 2 or len(contour_points) < 6:
            return None
        
        return contour_points.astype(np.float32)

    def direct_3p_fit(self, points: np.ndarray) -> np.ndarray:
        """
        Algebraic solution to find sphere center from contour points.
        Based on Tóth & Hajder (2023).
        """
        # Get normalized rays
        norm_pts = self.normalize_points(points)
        # Convert to unit vectors
        q_vectors = norm_pts / np.linalg.norm(norm_pts, axis=1, keepdims=True)
        Q = q_vectors.T  # Shape (3, N)
        
        # Solve (cos a)^-1 * w = Q^{-T} * 1
        # Use pseudoinverse for overdetermined system (>3 points)
        Q_pinv_T = np.linalg.pinv(Q.T)
        ones = np.ones(len(points))
        
        term = Q_pinv_T @ ones
        
        # Recover cone axis (w) and angle (alpha)
        norm_term = np.linalg.norm(term)
        w = term / norm_term
        cos_alpha = 1.0 / norm_term
        
        # Calculate distance to sphere center: d = r / sin(alpha)
        # sin(alpha) = sqrt(1 - cos^2(alpha))
        sin_alpha = np.sqrt(1 - cos_alpha**2)
        dist = self.sphere_radius / sin_alpha
        
        # Sphere center s = dist * w
        s = dist * w
        return s
    
    def optimize_center(self, init_center: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
        """
        Non-linear optimization minimizing the difference between the 
        sphere radius and the distance from the center to each back-projected ray.
        """
        # Pre-compute unit rays for all contour points
        norm_pts = self.normalize_points(contour_points)
        rays = norm_pts / np.linalg.norm(norm_pts, axis=1, keepdims=True)
        
        def residuals(s):
            # Distance from point 's' to line defined by unit vector 'ray' passing through origin
            # dist = || s - (s . ray) * ray ||
            
            # Vectorized projection of s onto all rays
            # dot_products shape: (N,)
            dot_products = np.dot(rays, s) 
            
            # projections shape: (N, 3)
            projections = rays * dot_products[:, np.newaxis]
            
            # distances to lines (rays)
            distances_to_rays = np.linalg.norm(s - projections, axis=1)
            
            # We want these distances to equal the sphere radius
            # residual = dist - r
            return distances_to_rays - self.sphere_radius

        # Run Least Squares optimization
        res = least_squares(residuals, init_center, loss='soft_l1', f_scale=0.1)
        return res.x

    def process(self, image_path: str, optimize: bool = True) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None: 
            return None
        
        # 1. Detect
        points = self.detect_contour(image)
        if points is None: 
            return None
        
        # 2. Initial Estimate
        try:
            center = self.direct_3p_fit(points)
        except:
            return None
        
        # 3. Optimize
        if optimize:
            center = self.optimize_center(center, points)
            
        return center

# ============================================================================
# FILE & MATCHING UTILS
# ============================================================================

def get_target_frames(csv_path: str) -> Set[str]:
    """
    Reads your teammate's CSV (test_fnXX) and returns a set of 'XX' strings.
    """
    targets = set()
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}")
        return targets
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn_str = row.get('fn', '')
            # Extract number from 'test_fn118' -> '118'
            match = re.search(r'fn(\d+)', fn_str)
            if match:
                targets.add(match.group(1))
    return targets

def find_images(img_dir: str, target_frames: Set[str]) -> list:
    """
    Finds images in img_dir that match the target frame numbers.
    Returns list of (filename, frame_number_string).
    """
    found = []
    if not os.path.exists(img_dir):
        print(f"ERROR: Could not find {img_dir}")
        return found
        
    print(f"Scanning {img_dir} for frames: {sorted(list(target_frames))}")
    
    for fname in sorted(os.listdir(img_dir)):
        # 1. Check extension
        if not fname.lower().endswith(('.jpg', '.png', '.bmp')):
            continue
            
        # 2. Apply Camera Filter (e.g. "Dev2")
        if CAMERA_FILTER not in fname:
            continue
            
        # 3. Extract Frame Number
        match = re.search(r'fn(\d+)', fname)
        if match:
            num = match.group(1)
            if num in target_frames:
                found.append((fname, num))
                
    return found

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("--- STARTING CAMERA SPHERE DETECTION ---")
    
    # 1. Load Targets
    target_frames = get_target_frames(LIDAR_CSV)
    if not target_frames:
        print("No target frames found in CSV. Exiting.")
        return
    print(f"Loaded {len(target_frames)} target frames from CSV.")

    # 2. Find Images
    images_to_process = find_images(IMAGE_DIR, target_frames)
    if not images_to_process:
        print("No matching images found. Check filenames/paths.")
        return
    print(f"Found {len(images_to_process)} matching images.")

    # 3. Process
    calib = CameraCalibration(CAMERA_MATRIX, SPHERE_RADIUS)
    results = []
    
    for fname, fnum in images_to_process:
        path = os.path.join(IMAGE_DIR, fname)
        print(f"Processing {fname}...", end="")
        
        center = calib.process(path, optimize=OPTIMIZE)
        
        if center is not None:
            # Format: fn, cx, cy, cz
            # Note: We reconstruct 'test_fnXX' to match teammate's format
            row_id = f"test_fn{fnum}"
            results.append([row_id, center[0], center[1], center[2]])
            print(f" OK -> {center}")
        else:
            print(f" FAILED (No contour?)")

    # 4. Save Output
    if results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['fn', 'cx', 'cy', 'cz']) # Header
            writer.writerows(results)
        print(f"\nSuccessfully saved {len(results)} rows to {OUTPUT_CSV}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()