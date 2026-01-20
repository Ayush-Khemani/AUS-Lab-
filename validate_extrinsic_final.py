# import numpy as np
# import pandas as pd
#
# # 1. Load FINAL aligned data
# lidar = pd.read_csv("outputs/centers_valid.csv")
# cam   = pd.read_csv("outputs/centers_cam.csv")
#
# # 按 fn 排序，确保一一对应（非常重要）
# lidar = lidar.sort_values("fn").reset_index(drop=True)
# cam   = cam.sort_values("fn").reset_index(drop=True)
#
# assert list(lidar["fn"]) == list(cam["fn"]), "Frame names do not match!"
#
# # 2. Extrinsic from compute_extrinsic.py (latest result)
# R = np.array([
#     [-0.36242813, -0.56394982, -0.7420286 ],
#     [ 0.9282382 , -0.28998422, -0.2329871 ],
#     [-0.08378355, -0.77322038,  0.62857821]
# ])
#
# t = np.array([0.04228157, 0.07178266, 0.21853596])
#
# errors = []
#
# print("Frame-wise reprojection error (LiDAR → Camera):")
# for i in range(len(lidar)):
#     p_l = lidar.loc[i, ["cx", "cy", "cz"]].values
#     p_c = cam.loc[i,   ["cx", "cy", "cz"]].values
#
#     p_pred = R @ p_l + t
#     err = np.linalg.norm(p_pred - p_c)
#     errors.append(err)
#
#     print(f"{lidar.loc[i,'fn']}: {err:.3f} m")
#
# errors = np.array(errors)
#
# print("\nSummary:")
# print(f"Mean error: {errors.mean():.3f} m")
# print(f"Max  error: {errors.max():.3f} m")


import numpy as np
import pandas as pd

# 1. Load FINAL aligned data
lidar = pd.read_csv("outputs/centers_valid.csv")
cam = pd.read_csv("outputs/centers_cam.csv")

# Sort by fn to ensure one-to-one correspondence
lidar = lidar.sort_values("fn").reset_index(drop=True)
cam = cam.sort_values("fn").reset_index(drop=True)

assert list(lidar["fn"]) == list(cam["fn"]), "Frame names do not match!"

# 2. Optimized Parameters from compute_extrinsic_advanced.py
# Use the results you just obtained:
s = 0.595188  # The scale factor s

R = np.array([
    [-0.36242813, 0.9282382, -0.08378357],
    [-0.56394981, -0.28998423, -0.77322038],
    [-0.74202861, -0.23298709, 0.6285782]
])

t = np.array([-0.02652915, 0.14934367, -0.03971506])

errors = []

print("Frame-wise verification error (Camera --(s,R,t)--> LiDAR):")
for i in range(len(lidar)):
    p_l = lidar.loc[i, ["cx", "cy", "cz"]].values
    p_c = cam.loc[i, ["cx", "cy", "cz"]].values

    # Transformation logic: s * R * p_camera + t
    # This transforms Kumar's coordinates into YOUR LiDAR world
    p_lidar_pred = s * (R @ p_c) + t

    # Calculate Euclidean distance between the predicted position and your LiDAR measurement
    err = np.linalg.norm(p_lidar_pred - p_l)
    errors.append(err)

    print(f"{lidar.loc[i, 'fn']}: {err:.3f} m")

errors = np.array(errors)

print("\n--- Final Summary ---")
print(f"Mean Error: {errors.mean():.6f} m")
print(f"Max Error : {errors.max():.6f} m")

if errors.mean() < 0.30:
    print("\nSUCCESS: The scale factor successfully reduced the calibration error.")