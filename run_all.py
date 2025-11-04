import glob
import csv
from pathlib import Path
import numpy as np
from lidar_sphere import fit_sphere_ransac, RansacConfig, load_xyz, save_result_json

def batch(cloud_glob="Data/Cld/test_fn*.xyz", r_known=0.25, out_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = [("fn", "cx", "cy", "cz", "radius", "inliers", "total")]
    files = sorted(glob.glob(cloud_glob))
    print(f"找到 {len(files)} 个点云文件。开始检测...")
    for path in files:
        try:
            pts = load_xyz(path)
            cfg = RansacConfig(
                max_iters=4000,
                inlier_tol=0.015,
                min_inliers=max(30, int(0.02 * len(pts))),
                r_known=r_known,
                refine=True,
            )
            c, r, inliers = fit_sphere_ransac(pts, cfg)
            fn = Path(path).stem
            save_result_json(f"{out_dir}/{fn}_sphere.json", c, r, int(inliers.sum()), int(len(pts)))
            rows.append((fn, c[0], c[1], c[2], r, int(inliers.sum()), int(len(pts))))
            print(f"✔ {fn}: center={c}, r={r:.4f}, inliers={inliers.sum()}/{len(pts)}")
        except Exception as e:
            print(f"✖ {path}: {e}")

    csv_path = f"{out_dir}/centers.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"所有结果已保存到 {csv_path}")

if __name__ == "__main__":
    batch()
