import numpy as np

# 读取原始点云
pts = np.loadtxt("Data/Cld/test_fn10.xyz")[:, :3]

# 选择坐标范围（这个范围是根据你截图中球的点坐标设定的）
mask = (
    (pts[:, 0] > 0.45) & (pts[:, 0] < 0.65) &
    (pts[:, 1] > 0.55) & (pts[:, 1] < 0.7) &
    (pts[:, 2] > -0.3) & (pts[:, 2] < 0.1)
)

roi = pts[mask]
print("原始点数:", len(pts))
print("ROI 点数:", len(roi))

# 保存筛选后的点云
np.savetxt("roi.xyz", roi)
print("已保存为 roi.xyz")
