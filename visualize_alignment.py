import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def project_lidar_to_image(pts_lidar, s, R, t, K):
    """
    Project LiDAR 3D points to Camera 2D pixel coordinates.
    Logic: p_cam = (1/s) * R^T * (p_lidar - t)
    """
    # 1. Transform to Camera Coordinate System
    # R is the rotation from Cam to Lidar, so use R.T for Lidar to Cam
    pts_cam = (1.0 / s) * (R.T @ (pts_lidar - t).T).T

    # 2. Project to Image Plane (u, v)
    # Using Camera Intrinsics K
    u_v_homogeneous = (K @ pts_cam.T).T

    # Normalize by Z-coordinate (Depth)
    z = u_v_homogeneous[:, 2]
    u = u_v_homogeneous[:, 0] / (z + 1e-6)
    v = u_v_homogeneous[:, 1] / (z + 1e-6)

    return u, v, z


if __name__ == "__main__":
    # --- [Step 1: Calibration Parameters from your latest run] ---
    s = 0.430872
    R = np.array([
        [0.17024539, 0.980902, -0.09406264],
        [-0.80422816, 0.08315061, -0.58847518],
        [-0.56941512, 0.17583301, 0.80302502]
    ])
    t = np.array([0.07864763, 0.05797798, -0.06835637])

    # Camera Intrinsics (from your lab data)
    # fu=625, fv=625, u0=480, v0=300
    K = np.array([
        [625, 0, 480],
        [0, 625, 300],
        [0, 0, 1]
    ])

    # --- [Step 2: Load Data for a specific frame, e.g., fn41] ---
    frame_id = "41"
    img_path = f"Data/Img/Dev0_Image_w960_h600_fn{frame_id}.jpg"
    lidar_path = f"Data/Cld/test_fn{frame_id}.xyz"

    if not os.path.exists(img_path) or not os.path.exists(lidar_path):
        print("Error: Missing image or point cloud file!")
        exit()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load LiDAR points (only the first 3 columns x,y,z)
    pts_lidar = np.loadtxt(lidar_path)[:, :3]

    # --- [Step 3: Projection] ---
    u, v, z = project_lidar_to_image(pts_lidar, s, R, t, K)

    # Filter points: only keep points in front of camera and within image boundaries
    h, w, _ = img.shape
    mask = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid, v_valid, z_valid = u[mask], v[mask], z[mask]

    # --- [Step 4: Visualize] ---
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Color points by depth for better visualization
    scatter = plt.scatter(u_valid, v_valid, c=z_valid, s=2, cmap='jet', alpha=0.5)
    plt.colorbar(scatter, label='Depth (m)')

    plt.title(f"Reprojection Visualization - Frame {frame_id}\nMean Error: 0.1456m")
    plt.axis('off')
    plt.show()