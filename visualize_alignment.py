import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 1. 读取数据 =====
lidar = pd.read_csv("outputs/centers_valid.csv")
cam   = pd.read_csv("outputs/centers_cam.csv")

# 确保按帧名对齐
common = sorted(set(lidar['fn']) & set(cam['fn']))
lidar = lidar[lidar['fn'].isin(common)].sort_values('fn')
cam   = cam[cam['fn'].isin(common)].sort_values('fn')

# LiDAR 球心
P_lidar = lidar[['cx', 'cy', 'cz']].to_numpy()
P_cam   = cam[['cx', 'cy', 'cz']].to_numpy()

# ===== 2. 填入你的外参 =====
R = np.array([
    [-0.36242813, -0.56394982, -0.7420286 ],
    [ 0.9282382 , -0.28998422, -0.2329871 ],
    [-0.08378355, -0.77322038,  0.62857821]
])

t = np.array([0.04228157, 0.07178266, 0.21853596])

# ===== 3. LiDAR → Camera =====
P_lidar_cam = (R @ P_lidar.T).T + t

# ===== 4. 3D 可视化 =====
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    P_cam[:, 0], P_cam[:, 1], P_cam[:, 2],
    c='red', s=60, label='Camera sphere centers'
)

ax.scatter(
    P_lidar_cam[:, 0], P_lidar_cam[:, 1], P_lidar_cam[:, 2],
    c='blue', s=60, marker='^', label='LiDAR → Camera'
)

# 连线（误差）
for i in range(len(P_cam)):
    ax.plot(
        [P_cam[i, 0], P_lidar_cam[i, 0]],
        [P_cam[i, 1], P_lidar_cam[i, 1]],
        [P_cam[i, 2], P_lidar_cam[i, 2]],
        'k--', linewidth=1
    )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("LiDAR–Camera Sphere Center Alignment")
ax.legend()
ax.view_init(elev=20, azim=120)

plt.tight_layout()
plt.show()
