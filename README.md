# LiDAR–Camera Calibration Project (AUS Lab)

This repository is part of the **AUS Laboratory Project**, focusing on the calibration between LiDAR and camera sensors using **sphere-based target calibration**.

The project implements and extends methods from the **ICRA 2020** and **IJCV 2023** papers, aiming to develop a **Python framework** for target-based LiDAR–camera calibration.

---

## Overview

###  Research Focus
- Implement **target-based calibration** methods between LiDAR and camera sensors.  
- Use **spherical targets** with known geometry to estimate the relative transformation (rotation + translation) between sensors.  
- Develop both **LiDAR-side** and **image-side** algorithms:
  - **LiDAR side:** Sphere detection and localization in 3D point clouds (Lee)
  - **Camera side:** Image-based circle/ellipse fitting (Kumar)
  - **Fusion:** Compute relative pose between LiDAR and camera

---

## 📁 Project Structure
```
LidarCalibration/
├── lidar_sphere.py # RANSAC-based sphere fitting for LiDAR point clouds
├── extract_roi.py # Extracts region of interest from raw point cloud
├── run_all.py # Batch process multiple LiDAR frames
├── filter_valid.py # Filter valid frames with successful sphere detection
├── compute_extrinsic.py # Compute LiDAR–camera extrinsic calibration
├── outputs/ # JSON/CSV results of detection and calibration
├── Data/ # Raw point clouds (.xyz) and images (.jpg)
├── .gitignore # Ignore unnecessary files (e.g., large data)
└── README.md
```