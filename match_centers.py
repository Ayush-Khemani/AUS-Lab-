import pandas as pd

lidar = pd.read_csv("outputs/centers_valid.csv")
cam = pd.read_csv("outputs/centers_cam.csv")

# 取交集
common = set(lidar['fn']).intersection(set(cam['fn']))

print("共同帧数：", len(common))
print("共同帧：", sorted(common))

# 过滤
lidar_matched = lidar[lidar['fn'].isin(common)].sort_values("fn")
cam_matched = cam[cam['fn'].isin(common)].sort_values("fn")

# 保存
lidar_matched.to_csv("outputs/centers_lidar_matched.csv", index=False)
cam_matched.to_csv("outputs/centers_cam_matched.csv", index=False)

print("\n匹配结果已保存为：")
print("  outputs/centers_lidar_matched.csv")
print("  outputs/centers_cam_matched.csv")
