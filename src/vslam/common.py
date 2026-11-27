import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def make_K(camera):#内参矩阵
    # fx、fy 是焦距，cx、cy 是主点坐标。
    return np.array([[camera["fx"], 0, camera["cx"]],
                     [0, camera["fy"], camera["cy"]],
                     [0, 0, 1]], dtype=np.float64)


def make_T(R, t):# 齐次变换矩阵
    # 将 R、t 组合成 4×4 齐次变换矩阵。
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return T


def transform(T, points):# 对一组3D点进行刚体变换
    # 对 Nx3 点集执行 p' = R p + t。
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def project(K, points):#
    # 针孔模型：相机三维坐标投影到像素平面。
    uvh = points @ K.T
    return uvh[:, :2] / uvh[:, 2:3]


def backproject(uv, z, K):# 反投影
    # 像素坐标乘 K^-1 得到视线方向，再乘深度得到三维点。
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K).T
    return rays * np.asarray(z)[:, None]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_tum(path, timestamps, poses_wc):
    rows = []
    for ts, T in zip(timestamps, poses_wc):
        # SciPy 输出四元数顺序为 xyzw，正好符合 TUM 格式。
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        rows.append([ts, *T[:3, 3], *q])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.9f")


def read_tum(path):
    # TUM 每行：timestamp tx ty tz qx qy qz qw。
    rows = np.loadtxt(path, comments="#", ndmin=2)
    poses = np.array([
        make_T(Rotation.from_quat(row[4:8]).as_matrix(), row[1:4])
        for row in rows
    ])
    return rows[:, 0], poses


def nearest_pose(ts, times, poses, tolerance=0.02):
    # 找最近时间戳；时间差超过 tolerance 秒就不关联。
    index = np.argmin(np.abs(times - ts))
    return poses[index] if abs(times[index] - ts) <= tolerance else None


def write_ply(path, points, colors=None):
    # 保存 ASCII PLY；未提供颜色时统一使用灰色。
    points = np.asarray(points)
    if colors is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")