import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.linalg import svd as scipy_svd
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class CameraCalibration:
    """
    Camera-based sphere calibration using ellipse detection and fitting.
    Based on papers:
    - "A Minimal Solution for Image-Based Sphere Estimation" (Tóth & Hajder, 2023)
    - "Automatic LiDAR-Camera Calibration" (Tóth et al., 2020)
    """
    
    def __init__(self, K: np.ndarray, sphere_radius: float):
        """
        Initialize camera calibration.
        
        Args:
            K: Camera intrinsic matrix (3x3)
            sphere_radius: Known radius of the calibration sphere (in meters)
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.sphere_radius = sphere_radius
        
    def normalize_image_coordinates(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize image coordinates using camera intrinsics.
        Converts pixel coordinates to normalized coordinates.
        
        Args:
            points: Array of shape (N, 2) with pixel coordinates [u, v]
            
        Returns:
            Array of shape (N, 3) with normalized homogeneous coordinates [û, v̂, 1]
        """
        if points.shape[1] != 2:
            raise ValueError("Points must be (N, 2) array")
            
        # Add homogeneous coordinate
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 3)
        
        # Normalize: [û, v̂, 1] = K^{-1} @ [u, v, 1]
        normalized = (self.K_inv @ points_h.T).T  # (N, 3)
        
        return normalized
    
    def detect_ellipse_contour(self, image: np.ndarray, 
                              canny_low: int = 50, 
                              canny_high: int = 150) -> Optional[np.ndarray]:
        """
        Detect ellipse contour points in image using Canny edge detection.
        
        Args:
            image: Input image (grayscale or color)
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            
        Returns:
            Array of detected contour points or None if no ellipse found
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Get largest contour (assuming it's the sphere)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert to array of points
        contour_points = largest_contour.squeeze()
        
        # Filter out degenerate cases
        if len(contour_points.shape) == 1 or len(contour_points) < 5:
            return None
        
        return contour_points.astype(np.float32)
    
    def fit_ellipse_from_points(self, points: np.ndarray) -> dict:
        """
        Fit ellipse to points using parametric representation.
        Implements the 3pFit algorithm from paper.
        
        Args:
            points: Contour points in pixel coordinates (N, 2)
            
        Returns:
            Dictionary with ellipse parameters: {a, b, e0, theta, coeffs}
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit ellipse")
        
        # Normalize coordinates
        normalized_pts = self.normalize_image_coordinates(points)  # (N, 3)
        
        # Create matrix Q with unit vectors
        q_vectors = normalized_pts / np.linalg.norm(normalized_pts, axis=1, keepdims=True)
        Q = q_vectors.T  # (3, N)
        
        # Compute cone parameters using minimal or overdetermined case
        if len(points) == 3:
            # Minimal case: (cos α)^{-1} w = Q^{-T} 1_3
            Q_inv_T = np.linalg.inv(Q.T)
            ones = np.ones(3)
        else:
            # Overdetermined case: (cos α)^{-1} w = Q^{+T} 1_n
            Q_pinv_T = np.linalg.pinv(Q.T)
            ones = np.ones(len(points))
            Q_inv_T = Q_pinv_T
        
        cos_alpha_inv_w = Q_inv_T @ ones
        w = cos_alpha_inv_w / np.linalg.norm(cos_alpha_inv_w)
        cos_alpha = 1.0 / np.linalg.norm(cos_alpha_inv_w)
        
        # Compute major axis endpoints using axial constraint
        a1, a2 = self._major_axis_endpoints(w, cos_alpha)
        
        # Compute ellipse center
        e0 = (a1 + a2) / 2
        
        # Semi-major axis
        a = np.linalg.norm(a1 - a2) / 2
        
        # Projectional constraint: compute semi-minor axis
        ex, ey = e0[0], e0[1]
        discriminant = ex**2 + ey**2 + 1 - a**2
        b = np.sqrt(-discriminant + np.sqrt(discriminant**2 + 4*a**2))
        
        # Rotation angle
        theta = np.arctan2(ey, ex)
        
        # Also compute implicit form coefficients
        ellipse_coeffs = self._parametric_to_implicit(e0, a, b, theta)
        
        return {
            'a': a,
            'b': b,
            'e0': e0,
            'theta': theta,
            'coeffs': ellipse_coeffs,
            'cos_alpha': cos_alpha,
            'w': w,
            'points': points
        }
    
    def _major_axis_endpoints(self, w: np.ndarray, cos_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute major axis endpoints from cone parameters."""
        p = np.array([0, 0, 1])  # Principal point in normalized coords
        
        # Compute normal to plane containing c, p, major axis
        cross = np.cross(p, w)
        if np.linalg.norm(cross) < 1e-6:
            n = np.array([0, 1, 0])
        else:
            n = cross / np.linalg.norm(cross)
        
        # Rotation angle
        alpha = np.arccos(np.clip(cos_alpha, -1, 1))
        
        # Rotation matrices for axis endpoints
        def rotate_axis_angle(v, axis, angle):
            """Rodrigues' rotation formula"""
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            return v + np.sin(angle) * (K @ v) + (1 - np.cos(angle)) * (K @ K @ v)
        
        q_a1 = rotate_axis_angle(w, n, alpha)
        q_a2 = rotate_axis_angle(w, n, -alpha)
        
        # Project to image plane
        a1 = q_a1[:2] / q_a1[2] if q_a1[2] != 0 else q_a1[:2]
        a2 = q_a2[:2] / q_a2[2] if q_a2[2] != 0 else q_a2[:2]
        
        return a1, a2
    
    def _parametric_to_implicit(self, e0: np.ndarray, a: float, b: float, theta: float) -> np.ndarray:
        """Convert parametric ellipse to implicit form coefficients [A, B, C, D, E, F]."""
        ex, ey = e0
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Standard form coefficients
        A = b**2 * cos_t**2 + a**2 * sin_t**2
        B = 2 * (a**2 - b**2) * sin_t * cos_t
        C = b**2 * sin_t**2 + a**2 * cos_t**2
        D = -2*A*ex - B*ey
        E = -B*ex - 2*C*ey
        F = A*ex**2 + B*ex*ey + C*ey**2 - a**2*b**2
        
        return np.array([A, B, C, D, E, F])
    
    def ellipse_to_sphere(self, ellipse_params: dict) -> np.ndarray:
        """
        Convert ellipse parameters to sphere center (Ell2Sphere algorithm).
        Requires known sphere radius.
        
        Args:
            ellipse_params: Dictionary from fit_ellipse_from_points
            
        Returns:
            Sphere center in 3D [x0, y0, z0] in camera coordinates
        """
        a1, a2 = self._major_axis_endpoints(
            ellipse_params['w'], 
            ellipse_params['cos_alpha']
        )
        
        # Back-project to 3D
        normalized_a1 = self.normalize_image_coordinates(a1.reshape(1, 2))[0]
        normalized_a2 = self.normalize_image_coordinates(a2.reshape(1, 2))[0]
        
        q_a1 = normalized_a1 / np.linalg.norm(normalized_a1)
        q_a2 = normalized_a2 / np.linalg.norm(normalized_a2)
        
        # Cone opening angle
        dot_product = np.clip(np.dot(q_a1, q_a2), -1, 1)
        cos_2alpha = dot_product
        
        # Sphere center
        s = np.sqrt(2 * self.sphere_radius**2 / (1 - cos_2alpha)) * (q_a1 + q_a2) / 2
        
        return s
    
    def direct_3p_fit(self, points: np.ndarray) -> np.ndarray:
        """
        Direct sphere center estimation from contour points (Direct3pFit algorithm).
        
        Args:
            points: Contour points in pixel coordinates (N, 2), N >= 3
            
        Returns:
            Sphere center in 3D [x0, y0, z0]
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points")
        
        # Normalize coordinates
        normalized_pts = self.normalize_image_coordinates(points)
        q_vectors = normalized_pts / np.linalg.norm(normalized_pts, axis=1, keepdims=True)
        Q = q_vectors.T  # (3, N)
        
        # Compute cone parameters
        if len(points) == 3:
            Q_inv_T = np.linalg.inv(Q.T)
        else:
            Q_inv_T = np.linalg.pinv(Q.T)
        
        cos_alpha_inv_w = Q_inv_T @ np.ones(len(points))
        w = cos_alpha_inv_w / np.linalg.norm(cos_alpha_inv_w)
        cos_alpha = 1.0 / np.linalg.norm(cos_alpha_inv_w)
        
        # Sphere center from cone parameters
        s = (self.sphere_radius / np.sqrt(1 - cos_alpha**2)) * w
        
        return s
    
    def optimize_sphere_center(self, sphere_center_init: np.ndarray, 
                              contour_points: np.ndarray,
                              max_iterations: int = 100) -> np.ndarray:
        """
        Optimize sphere center using Levenberg-Marquardt algorithm.
        
        Args:
            sphere_center_init: Initial sphere center estimate
            contour_points: Detected contour points in pixel coordinates
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized sphere center
        """
        def cost_function(s_flat):
            s = s_flat.reshape(3)
            # Project sphere to image and compute distance to contour
            error = 0
            
            for pt in contour_points:
                # Expected projection
                normalized_pt = self.normalize_image_coordinates(pt.reshape(1, 2))[0]
                ray = normalized_pt / np.linalg.norm(normalized_pt)
                
                # Distance from sphere surface along ray
                # Solve: |z*ray - s|^2 = r^2 for z
                a_coef = np.dot(ray, ray)
                b_coef = -2 * np.dot(ray, s)
                c_coef = np.dot(s, s) - self.sphere_radius**2
                
                discriminant = b_coef**2 - 4*a_coef*c_coef
                if discriminant >= 0:
                    z = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)
                    if z > 0:
                        proj_pt = z * ray
                        error += np.linalg.norm(proj_pt - s)**2
            
            return error
        
        result = minimize(cost_function, sphere_center_init.flatten(), 
                         method='Nelder-Mead',
                         options={'maxiter': max_iterations, 'xatol': 1e-8})
        
        return result.x.reshape(3)
    
    def process_image(self, image_path: str, optimize: bool = True) -> Optional[dict]:
        """
        Complete pipeline: detect sphere in image and estimate center.
        
        Args:
            image_path: Path to image file
            optimize: Whether to optimize sphere center
            
        Returns:
            Dictionary with results or None if failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Detect ellipse contour
        contour_points = self.detect_ellipse_contour(image)
        if contour_points is None or len(contour_points) < 5:
            print(f"Warning: Could not detect sufficient contour points in {image_path}")
            return None
        
        # Fit ellipse
        ellipse_params = self.fit_ellipse_from_points(contour_points)
        
        # Estimate sphere center
        sphere_center = self.direct_3p_fit(contour_points)
        
        # Optimize if requested
        if optimize:
            sphere_center = self.optimize_sphere_center(sphere_center, contour_points)
        
        return {
            'image_path': image_path,
            'contour_points': contour_points,
            'ellipse_params': ellipse_params,
            'sphere_center': sphere_center,
            'image': image
        }


class MultiImageCalibration:
    """
    Calibration using multiple images and point cloud data.
    """
    
    def __init__(self, K: np.ndarray, sphere_radius: float):
        """Initialize with camera intrinsics."""
        self.camera = CameraCalibration(K, sphere_radius)
        self.sphere_radius = sphere_radius
        
    def load_point_cloud(self, xyz_file: str) -> Optional[np.ndarray]:
        """
        Load point cloud from .xyz file (LiDAR data).
        
        Args:
            xyz_file: Path to .xyz file
            
        Returns:
            Array of 3D points or None if failed
        """
        try:
            points = np.loadtxt(xyz_file, delimiter=' ')
            if len(points.shape) == 1:
                points = points.reshape(1, -1)
            return points[:, :3]  # Take only x, y, z columns
        except Exception as e:
            print(f"Error loading point cloud {xyz_file}: {e}")
            return None
    
    def process_image_sequence(self, image_dir: str) -> List[dict]:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            List of processing results
        """
        results = []
        
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith(('.jpg', '.png', '.bmp')):
                image_path = os.path.join(image_dir, filename)
                result = self.camera.process_image(image_path)
                if result is not None:
                    results.append(result)
        
        return results
    
    def process_point_cloud_sequence(self, cloud_dir: str) -> List[dict]:
        """
        Process all point clouds in a directory.
        
        Args:
            cloud_dir: Directory containing .xyz files
            
        Returns:
            List of point cloud data
        """
        results = []
        
        for filename in sorted(os.listdir(cloud_dir)):
            if filename.endswith('.xyz'):
                cloud_path = os.path.join(cloud_dir, filename)
                points = self.load_point_cloud(cloud_path)
                if points is not None:
                    results.append({
                        'filename': filename,
                        'points': points,
                        'path': cloud_path
                    })
        
        return results


# Example usage
if __name__ == "__main__":
    # Camera parameters (from your specification)
    K = np.array([
        [625, 0, 480],
        [0, 625, 300],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Sphere radius in meters
    sphere_radius = 0.25
    
    # Initialize calibration
    calibration = CameraCalibration(K, sphere_radius)
    
    # Example: Process a single image
    # image_path = "path/to/image.jpg"
    # result = calibration.process_image(image_path)
    # if result:
    #     print(f"Sphere center: {result['sphere_center']}")
    
    # Example: Process sequence
    multi_calib = MultiImageCalibration(K, sphere_radius)
    
    # Process images
    # image_results = multi_calib.process_image_sequence("path/to/images")
    
    # Process point clouds
    # cloud_results = multi_calib.process_point_cloud_sequence("path/to/Cld")
    
    print("Camera calibration module ready for use.")