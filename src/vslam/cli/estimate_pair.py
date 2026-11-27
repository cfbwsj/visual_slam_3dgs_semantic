import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vslam.common import project, transform, write_json, write_ply
from vslam.geometry.two_view import estimate_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # np.load 读取阶段一保存的匹配点和相机参数。
    with np.load(args.input) as data:
        result, report = estimate_pair(
            data["points0"], data["points1"], data["K"], data["distortion"]
        )
        for key in ["timestamp0", "timestamp1", "index0", "index1"]:
            result[key] = data[key].copy()

    np.savez_compressed(out / "geometry.npz", **result)
    write_ply(out / "points_c0.ply", result["points_c0"])
    write_json(out / "report.json", report)

    # 把三维点变换到第二帧，再投影回图像，检查重投影是否贴合观测点。
    points_c1 = transform(result["T_c1_c0"], result["points_c0"])
    projected = project(result["K"], points_c1)
    observed = result["uv1"]

    fig, ax = plt.subplots()
    ax.scatter(observed[:, 0], observed[:, 1], s=4, label="observed")
    ax.scatter(projected[:, 0], projected[:, 1], s=4, label="reprojected")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(out / "reprojection.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()