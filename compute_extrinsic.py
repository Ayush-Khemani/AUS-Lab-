# import numpy as np
# import csv
#
# def load_centers(csv_path):
#     data = {}
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             fn = row["fn"]
#             try:
#                 c = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])])
#                 data[fn] = c
#             except Exception:
#                 continue
#     return data
#
# def compute_rigid_transform(A, B):
#     """
#     使用 SVD 求解刚体变换: B ≈ R*A + t
#     A: N×3 LiDAR 坐标点
#     B: N×3 Camera 坐标点
#     """
#     assert A.shape == B.shape
#
#     centroid_A = A.mean(axis=0)
#     centroid_B = B.mean(axis=0)
#
#     AA = A - centroid_A
#     BB = B - centroid_B
#
#     H = AA.T @ BB
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
#
#     # 保证旋转矩阵右手性（det(R)=1）
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = Vt.T @ U.T
#
#     t = centroid_B - R @ centroid_A
#     return R, t
#
# if __name__ == "__main__":
#     lidar_csv = "outputs/centers_valid.csv"
#     cam_csv = "outputs/centers_cam.csv"  # Kumar 的结果
#
#     lidar = load_centers(lidar_csv)
#     cam = load_centers(cam_csv)
#
#     # 找两边共有的帧
#     common = sorted(set(lidar.keys()) & set(cam.keys()))
#     if len(common) < 3:
#         raise ValueError(f"帧匹配太少（找到 {len(common)} 个）。至少需要 3 个对应球心。")
#
#     A = np.array([lidar[k] for k in common])
#     B = np.array([cam[k] for k in common])
#
#     print(f"使用 {len(common)} 对匹配帧计算外参:")
#     for k in common:
#         print(" ", k)
#
#     R, t = compute_rigid_transform(A, B)
#
#     print("\n旋转矩阵 R =")
#     print(R)
#     print("\n平移向量 t =")
#     print(t)
#
#     # 检查拟合误差
#     A_transformed = (R @ A.T).T + t
#     error = np.linalg.norm(A_transformed - B, axis=1)
#     print(f"\n平均误差: {error.mean():.6f} m, 最大误差: {error.max():.6f} m")
#
#


# import numpy as np
# import csv
# import os
# from scipy.optimize import least_squares
# from scipy.spatial.transform import Rotation as R_tool
#
# def load_centers(csv_path):
#     """
#     Load sphere centers from a CSV file.
#     """
#     data = {}
#     if not os.path.exists(csv_path):
#         return data
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             fn = row["fn"]
#             try:
#                 # Assume CSV contains cx, cy, cz columns
#                 c = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])])
#                 data[fn] = c
#             except Exception:
#                 continue
#     return data
#
# def compute_similitude_transform(A, B):
#     """
#     Closed-form solution: solve A ≈ s * R * B + t (7-DOF similitude transform).
#     A: LiDAR coordinates (N, 3) - Reference scale (meters)
#     B: Camera coordinates (N, 3) - Unknown/arbitrary scale
#     """
#     n = A.shape[0]
#     mu_a = A.mean(axis=0)
#     mu_b = B.mean(axis=0)
#
#     AA = A - mu_a
#     BB = B - mu_b
#
#     # Calculate scale s based on variance ratio
#     var_b = np.sum(BB ** 2) / n
#     H = (AA.T @ BB) / n
#     U, S, Vt = np.linalg.svd(H)
#
#     R = U @ Vt
#     # Ensure rotation matrix right-handedness (det(R)=1)
#     if np.linalg.det(R) < 0:
#         U[:, -1] *= -1
#         R = U @ Vt
#
#     # Optimal closed-form solution for scale s
#     s = np.trace(np.diag(S)) / var_b
#     t = mu_a - s * (R @ mu_b)
#
#     return s, R, t
#
# def residual_function(params, A, B):
#     """
#     Residual function for Bundle Adjustment (BA) optimization.
#     params: [s, qx, qy, qz, qw, tx, ty, tz]
#     """
#     s = params[0]
#     quat = params[1:5]
#     t = params[5:8]
#
#     rot = R_tool.from_quat(quat).as_matrix()
#     # Predicted coordinates: p_lidar = s * R * p_cam + t
#     B_transformed = s * (rot @ B.T).T + t
#     return (A - B_transformed).flatten()
#
# def refine_extrinsic_ba(s0, R0, t0, A, B):
#     """
#     Nonlinear refinement (Bundle Adjustment logic) using Levenberg-Marquardt.
#     """
#     # Initial parameter vector: scale + quaternion + translation
#     quat0 = R_tool.from_matrix(R0).as_quat()
#     params0 = np.hstack([[s0], quat0, t0])
#
#     # Optimize using Levenberg-Marquardt algorithm
#     res = least_squares(residual_function, params0, args=(A, B), method='lm')
#
#     s_opt = res.x[0]
#     R_opt = R_tool.from_quat(res.x[1:5]).as_matrix()
#     t_opt = res.x[5:8]
#     return s_opt, R_opt, t_opt
#
# if __name__ == "__main__":
#     # 1. Load sphere center data
#     lidar_csv = "outputs/centers_valid.csv"  # Lee's cleaned LiDAR data
#     cam_csv = "outputs/centers_cam.csv"      # Kumar's camera data
#
#     lidar_data = load_centers(lidar_csv)
#     cam_data = load_centers(cam_csv)
#
#     # Find common frames between both sensors
#     common = sorted(set(lidar_data.keys()) & set(cam_data.keys()))
#     if len(common) < 3:
#         print(f"Too few matching frames found: {len(common)}")
#         exit()
#
#     A = np.array([lidar_data[k] for k in common])  # Target LiDAR points
#     B = np.array([cam_data[k] for k in common])    # Input Camera points
#
#     print(f"--- Phase 1: 7-DOF Closed-form Solution (Handling Unknown Scale) ---")
#     s_init, R_init, t_init = compute_similitude_transform(A, B)
#
#     # Initial error check
#     err_init = np.linalg.norm(A - (s_init * (R_init @ B.T).T + t_init), axis=1)
#     print(f"Estimated scale s: {s_init:.6f}")
#     print(f"Initial mean error: {err_init.mean():.6f} m")
#
#     print(f"\n--- Phase 2: Nonlinear Refinement (BA Optimization) ---")
#     s_fin, R_fin, t_fin = refine_extrinsic_ba(s_init, R_init, t_init, A, B)
#
#     # Final error check
#     err_fin = np.linalg.norm(A - (s_fin * (R_fin @ B.T).T + t_fin), axis=1)
#
#     print(f"Optimized scale s: {s_fin:.6f}")
#     print(f"Optimized rotation matrix R:\n{R_fin}")
#     print(f"Optimized translation vector t: {t_fin}")
#     print(f"Final mean error: {err_fin.mean():.6f} m")
#     print(f"Maximum error: {err_fin.max():.6f} m")
#
#     # Verify task requirements: if s is significantly different from 1, scale was a major error source
#     if abs(s_fin - 1.0) > 0.05:
#         print(f"\nConclusion: Significant scale difference detected (s={s_fin:.2f}). "
#               f"The improved algorithm successfully compensated for the camera-side depth error.")


import numpy as np
import csv
import os
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_tool


# --- [Functions from previous steps remain the same] ---

def load_centers(csv_path):
    data = {}
    if not os.path.exists(csv_path): return data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data[row["fn"]] = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])])
            except:
                continue
    return data


def compute_similitude_transform(A, B):
    n = A.shape[0]
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - mu_a, B - mu_b
    var_b = np.sum(BB ** 2) / n
    H = (AA.T @ BB) / n
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = np.trace(np.diag(S)) / var_b
    t = mu_a - s * (R @ mu_b)
    return s, R, t


def residual_function(params, A, B):
    s, quat, t = params[0], params[1:5], params[5:8]
    rot = R_tool.from_quat(quat).as_matrix()
    return (A - (s * (rot @ B.T).T + t)).flatten()


def refine_extrinsic_ba(s0, R0, t0, A, B):
    params0 = np.hstack([[s0], R_tool.from_matrix(R0).as_quat(), t0])
    res = least_squares(residual_function, params0, args=(A, B), method='lm')
    return res.x[0], R_tool.from_quat(res.x[1:5]).as_matrix(), res.x[5:8]


# --- [New Robust Loop Logic] ---

if __name__ == "__main__":
    lidar_data = load_centers("outputs/centers_valid.csv")
    cam_data = load_centers("outputs/centers_cam.csv")

    common_frames = sorted(set(lidar_data.keys()) & set(cam_data.keys()))

    # Target: Mean error < 0.15m or at least 5 frames left
    ERROR_THRESHOLD = 0.15
    MIN_FRAMES = 5

    while len(common_frames) >= MIN_FRAMES:
        A = np.array([lidar_data[k] for k in common_frames])
        B = np.array([cam_data[k] for k in common_frames])

        # Step 1: Initial Guess
        s_i, R_i, t_i = compute_similitude_transform(A, B)
        # Step 2: BA Refinement
        s, R, t = refine_extrinsic_ba(s_i, R_i, t_i, A, B)

        # Step 3: Calculate per-frame errors
        A_pred = s * (R @ B.T).T + t
        frame_errors = np.linalg.norm(A - A_pred, axis=1)
        mean_err = frame_errors.mean()

        print(f"\nIteration with {len(common_frames)} frames:")
        print(f"  Mean Error: {mean_err:.4f} m")

        if mean_err < ERROR_THRESHOLD:
            print(">>> Success: Error threshold reached!")
            break

        # Step 4: Find the worst frame and remove it
        worst_idx = np.argmax(frame_errors)
        worst_frame = common_frames.pop(worst_idx)
        print(f"  Removing Outlier: {worst_frame} (Error: {frame_errors[worst_idx]:.4f} m)")

    print("\n--- FINAL OPTIMIZED EXTRINSICS ---")
    print(f"Scale s: {s:.6f}")
    print(f"Rotation R:\n{R}")
    print(f"Translation t: {t}")
    print(f"Final Mean Error: {mean_err:.6f} m")