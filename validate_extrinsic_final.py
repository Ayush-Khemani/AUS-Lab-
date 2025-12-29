import numpy as np
import pandas as pd

# 1. Load FINAL aligned data
lidar = pd.read_csv("outputs/centers_valid.csv")
cam   = pd.read_csv("outputs/centers_cam.csv")

# 按 fn 排序，确保一一对应（非常重要）
lidar = lidar.sort_values("fn").reset_index(drop=True)
cam   = cam.sort_values("fn").reset_index(drop=True)

assert list(lidar["fn"]) == list(cam["fn"]), "Frame names do not match!"

# 2. Extrinsic from compute_extrinsic.py (latest result)
R = np.array([
    [-0.36242813, -0.56394982, -0.7420286 ],
    [ 0.9282382 , -0.28998422, -0.2329871 ],
    [-0.08378355, -0.77322038,  0.62857821]
])

t = np.array([0.04228157, 0.07178266, 0.21853596])

errors = []

print("Frame-wise reprojection error (LiDAR → Camera):")
for i in range(len(lidar)):
    p_l = lidar.loc[i, ["cx", "cy", "cz"]].values
    p_c = cam.loc[i,   ["cx", "cy", "cz"]].values

    p_pred = R @ p_l + t
    err = np.linalg.norm(p_pred - p_c)
    errors.append(err)

    print(f"{lidar.loc[i,'fn']}: {err:.3f} m")

errors = np.array(errors)

print("\nSummary:")
print(f"Mean error: {errors.mean():.3f} m")
print(f"Max  error: {errors.max():.3f} m")
