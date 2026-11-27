import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vslam.common import read_tum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # TUM 文件读出的 poses 是 T_wc，平移部分就是相机在世界坐标中的位置。
    _, poses = read_tum(args.trajectory)
    xyz = poses[:, :3, 3]

    # Matplotlib 的 3D 坐标轴用于绘制相机运动轨迹。
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*xyz.T)
    ax.scatter(*xyz[0], label="start")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.legend()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()