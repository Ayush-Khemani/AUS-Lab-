"""
LiDAR sphere detection and localization module (Lee's scope)

Implements:
  - RANSAC-based sphere detection in point clouds (minimal 4-point solver)
  - Optional known-radius constraint (e.g., r=0.25 m) or free-radius estimation
  - Nonlinear refinement via Gauss-Newton least squares (center and radius)
  - Simple CLI entrypoint for batch processing frames

Dependencies: numpy

Usage (as a module):
    from lidar_sphere import fit_sphere_ransac
    center, radius, inliers = fit_sphere_ransac(points, r_known=0.25)

CLI examples:
    python lidar_sphere.py --cloud Data/Cld/test_fn10.xyz --r-known 0.25
    python lidar_sphere.py --cloud Data/Cld/test_fn10.xyz --save-json outputs/fn10_sphere.json

Author: Lee
"""
from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ------------------------------
# Math utilities
# ------------------------------

def sphere_from_4_points(P: np.ndarray) -> Tuple[np.ndarray, float]:
    """Closed-form sphere through 4 non-coplanar points.
    P: (4,3) array
    Returns: (center (3,), radius)
    """
    assert P.shape == (4, 3)
    # Solve for sphere in form ||x||^2 + a x + b y + c z + d = 0
    # For each point p: ||p||^2 + a px + b py + c pz + d = 0
    A = np.hstack([P, np.ones((4, 1))])  # [x y z 1]
    b = -np.sum(P**2, axis=1)            # -||p||^2
    # Solve linear system A*[a b c d]^T = b
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("Degenerate configuration for 4-point sphere")
    a, b_, c, d = x
    center = -0.5 * np.array([a, b_, c])
    R2 = np.dot(center, center) - d
    if R2 <= 0:
        raise ValueError("Computed negative radius^2; degenerate points")
    radius = float(math.sqrt(R2))
    return center, radius


def residuals_center_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    d = np.linalg.norm(points - center[None, :], axis=1)
    return d - radius


def gauss_newton_refine(
    points: np.ndarray,
    center0: np.ndarray,
    radius0: float,
    fix_radius: Optional[float] = None,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Refine center and (optionally) radius by minimizing sum((||p - c|| - r)^2).
    If fix_radius is not None, keeps r fixed to that value and only optimizes c.
    Returns: (center, radius)
    """
    c = center0.astype(float).copy()
    r = float(radius0)

    for _ in range(max_iters):
        v = points - c[None, :]
        d = np.linalg.norm(v, axis=1)
        # Prevent division by zero
        eps = 1e-12
        inv_d = 1.0 / np.maximum(d, eps)
        r_i = d - (fix_radius if fix_radius is not None else r)  # residuals

        # Jacobian J: each row [∂ri/∂cx, ∂ri/∂cy, ∂ri/∂cz, ∂ri/∂r]
        Jc = -v * inv_d[:, None]  # ∂(d)/∂c = -(p-c)/||p-c||
        if fix_radius is None:
            J = np.hstack([Jc, -np.ones((points.shape[0], 1))])
        else:
            J = Jc

        # Solve normal equations J^T J Δ = -J^T r
        try:
            JTJ = J.T @ J
            JTr = J.T @ r_i
            delta = -np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            # Fallback to least squares solver
            delta, *_ = np.linalg.lstsq(J, -r_i, rcond=None)

        if fix_radius is None:
            dc = delta[:3]
            dr = float(delta[3])
            c += dc
            r += dr
        else:
            c += delta[:3]

        if np.linalg.norm(delta[:3]) < tol and (fix_radius is not None or abs(delta[-1]) < tol):
            break

    if fix_radius is not None:
        r = float(fix_radius)
    return c, r


# ------------------------------
# RANSAC
# ------------------------------

@dataclass
class RansacConfig:
    max_iters: int = 2000
    inlier_tol: float = 0.01  # meters; tighten when using known radius
    min_inliers: int = 50
    r_known: Optional[float] = None  # e.g., 0.25
    refine: bool = True


def fit_sphere_ransac(points: np.ndarray, cfg: RansacConfig | None = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """RANSAC to detect a sphere in a point cloud.
    points: (N,3)
    Returns: (best_center, best_radius, inlier_mask)
    """
    if cfg is None:
        cfg = RansacConfig()
    N = points.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 points for sphere fitting")

    best_center = None
    best_radius = None
    best_inliers = None
    best_count = -1

    rng = np.random.default_rng(42)

    for _ in range(cfg.max_iters):
        # Sample 4 distinct indices
        idx = rng.choice(N, size=4, replace=False)
        P4 = points[idx]
        try:
            c0, r0 = sphere_from_4_points(P4)
        except Exception:
            continue
        # If known radius, reject early if far off
        if cfg.r_known is not None and abs(r0 - cfg.r_known) > 3 * cfg.inlier_tol:
            # Allow some slack; else discard
            pass

        # Compute residuals to candidate sphere
        d = np.linalg.norm(points - c0[None, :], axis=1)
        r_use = cfg.r_known if cfg.r_known is not None else r0
        res = np.abs(d - r_use)
        inliers = res < cfg.inlier_tol
        count = int(inliers.sum())

        if count > best_count and count >= cfg.min_inliers:
            best_center = c0
            best_radius = r_use if cfg.r_known is not None else r0
            best_inliers = inliers
            best_count = count

    if best_inliers is None or best_count < cfg.min_inliers:
        raise RuntimeError("RANSAC failed to find a consensus sphere (try loosening tol / increasing max_iters)")

    # Refine with inliers
    if cfg.refine:
        P_in = points[best_inliers]
        c_ref, r_ref = gauss_newton_refine(
            P_in, best_center, best_radius, fix_radius=cfg.r_known
        )
        best_center, best_radius = c_ref, r_ref

    return best_center, float(best_radius), best_inliers


# ------------------------------
# I/O utilities
# ------------------------------

def load_xyz(path: str) -> np.ndarray:
    """Load .xyz file (at least 3 columns). Ignores extra columns."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    return raw[:, :3]


def save_result_json(path: str, center: np.ndarray, radius: float, inlier_count: int, total: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "center": center.tolist(),
        "radius": float(radius),
        "inliers": int(inlier_count),
        "total": int(total),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="RANSAC sphere detection for LiDAR point clouds")
    ap.add_argument("--cloud", required=True, help="Path to .xyz point cloud")
    ap.add_argument("--r-known", type=float, default=None, help="Known sphere radius in meters (optional)")
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=0.01, help="Inlier tolerance (meters)")
    ap.add_argument("--min-inliers", type=int, default=50)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--save-json", default=None, help="Path to save result JSON")
    args = ap.parse_args()

    pts = load_xyz(args.cloud)
    cfg = RansacConfig(
        max_iters=args.max_iters,
        inlier_tol=args.tol,
        min_inliers=args.min_inliers,
        r_known=args.r_known,
        refine=not args.no_refine,
    )
    c, r, inliers = fit_sphere_ransac(pts, cfg)

    print(f"Center: {c}")
    print(f"Radius: {r:.6f} m")
    print(f"Inliers: {inliers.sum()} / {pts.shape[0]}")

    if args.save_json:
        save_result_json(args.save_json, c, r, int(inliers.sum()), int(pts.shape[0]))
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()


# ------------------------------
# Helper: batch process all frames and export CSV
# Save this snippet as run_all.py next to lidar_sphere.py
# ------------------------------
if False:
    import glob
    import csv
    from pathlib import Path

    def batch(cloud_glob: str = "Data/Cld/test_fn*.xyz", r_known: float | None = 0.25, out_dir: str = "outputs"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rows = [("fn", "cx", "cy", "cz", "radius", "inliers", "total")]
        for path in sorted(glob.glob(cloud_glob)):
            pts = load_xyz(path)
            cfg = RansacConfig(r_known=r_known, inlier_tol=0.01, max_iters=3000, min_inliers=max(30, int(0.02*len(pts))))
            try:
                c, r, inliers = fit_sphere_ransac(pts, cfg)
                fn = Path(path).stem.split("_fn")[-1]
                save_result_json(f"{out_dir}/fn{fn}_sphere.json", c, r, int(inliers.sum()), int(pts.shape[0]))
                rows.append((f"fn{fn}", c[0], c[1], c[2], r, int(inliers.sum()), int(pts.shape[0])))
                print(f"OK fn{fn}: center={c}, r={r:.4f}, inliers={inliers.sum()}/{pts.shape[0]}")
            except Exception as e:
                print(f"FAIL {path}: {e}")
        # write CSV summary
        with open(f"{out_dir}/centers.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"CSV saved to {out_dir}/centers.csv")

    # Uncomment to run directly
    # batch()
