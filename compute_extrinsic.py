import numpy as np
import csv

def load_centers(csv_path):
    data = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["fn"]
            try:
                c = np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])])
                data[fn] = c
            except Exception:
                continue
    return data

def compute_rigid_transform(A, B):
    """
    使用 SVD 求解刚体变换: B ≈ R*A + t
    A: N×3 LiDAR 坐标点
    B: N×3 Camera 坐标点
    """
    assert A.shape == B.shape

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 保证旋转矩阵右手性（det(R)=1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t

if __name__ == "__main__":
    lidar_csv = "outputs/centers_valid.csv"
    cam_csv = "outputs/centers_cam.csv"  # Kumar 的结果

    lidar = load_centers(lidar_csv)
    cam = load_centers(cam_csv)

    # 找两边共有的帧
    common = sorted(set(lidar.keys()) & set(cam.keys()))
    if len(common) < 3:
        raise ValueError(f"帧匹配太少（找到 {len(common)} 个）。至少需要 3 个对应球心。")

    A = np.array([lidar[k] for k in common])
    B = np.array([cam[k] for k in common])

    print(f"使用 {len(common)} 对匹配帧计算外参:")
    for k in common:
        print(" ", k)

    R, t = compute_rigid_transform(A, B)

    print("\n旋转矩阵 R =")
    print(R)
    print("\n平移向量 t =")
    print(t)

    # 检查拟合误差
    A_transformed = (R @ A.T).T + t
    error = np.linalg.norm(A_transformed - B, axis=1)
    print(f"\n平均误差: {error.mean():.6f} m, 最大误差: {error.max():.6f} m")
