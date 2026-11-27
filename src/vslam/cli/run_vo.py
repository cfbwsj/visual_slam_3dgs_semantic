import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from vslam.common import backproject, make_K, make_T, write_json, write_tum
from vslam.config import load_config
from vslam.data.association import read_associations
from vslam.vo.features import extract_orb, match_ratio, matched_points


def load_rgbd(root, row):
    # row 中第 2、4 项分别是 RGB 和深度图的相对路径。
    rgb = cv2.imread(str(root / row[1]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / row[3]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"读取图像失败：{row}")
    return rgb, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--associations", required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root, out = Path(cfg["dataset"]["root"]), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_associations(args.associations)
    if args.max_frames > 0:
        rows = rows[:args.max_frames]
    if len(rows) < 2:
        raise ValueError("至少需要两组 RGB-D 数据")

    K, vo = make_K(cfg["camera"]), cfg["vo"]
    if np.any(cfg["camera"]["distortion"]):
        raise ValueError("当前 RGB-D 基线使用零畸变模型")

    # 第一帧作为世界坐标系原点，初始相机位姿为单位矩阵。
    image0, depth0 = load_rgbd(root, rows[0])
    kp0, des0 = extract_orb(image0, cfg["orb"]["nfeatures"])
    poses, stamps, logs = [np.eye(4)], [rows[0][0]], []

    for row in rows[1:]:
        log = dict(timestamp=row[0], matches=0, inliers=0,
                   median_error_px=None, status="ok")
        try:
            image1, depth1 = load_rgbd(root, row)
            kp1, des1 = extract_orb(image1, cfg["orb"]["nfeatures"])

            # ORB 匹配得到两帧中的像素对应关系。
            matches = match_ratio(des0, des1, cfg["matching"]["ratio"])
            uv0, uv1 = matched_points(kp0, kp1, matches)
            log["matches"] = len(matches)

            # 从上一帧深度图读取深度，并反投影成三维点。
            pixels = np.rint(uv0).astype(int)
            z = depth0[pixels[:, 1], pixels[:, 0]].astype(float)
            z /= cfg["camera"]["depth_scale"]
            valid = (z >= vo["min_depth"]) & (z <= vo["max_depth"])
            points3d = backproject(uv0[valid], z[valid], K)
            points2d = uv1[valid]

            if len(points3d) < vo["min_correspondences"]:
                raise RuntimeError("有效深度对应点不足")

            # solvePnPRansac 用 3D-2D 对应关系估计当前帧相对位姿。
            ok, rvec, tvec, ids = cv2.solvePnPRansac(
                points3d, points2d, K, None,
                iterationsCount=200,
                reprojectionError=vo["reprojection_px"],
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP
            )
            if not ok or ids is None:
                raise RuntimeError("PnP 求解失败")

            ids = ids.ravel()
            log["inliers"] = len(ids)
            if (len(ids) < vo["min_inliers"]
                    or len(ids) / len(points3d) < vo["min_inlier_ratio"]):
                raise RuntimeError("PnP 内点不足")

            # solvePnPRefineLM 在 RANSAC 内点上细化位姿。
            rvec, tvec = cv2.solvePnPRefineLM(
                points3d[ids], points2d[ids], K, None, rvec, tvec
            )
            projected, _ = cv2.projectPoints(points3d[ids], rvec, tvec, K, None)
            error = np.linalg.norm(projected.reshape(-1, 2) - points2d[ids], axis=1)
            log["median_error_px"] = float(np.median(error))

            if log["median_error_px"] > vo["reprojection_px"]:
                raise RuntimeError("重投影误差过大")

            # PnP 输出 T_c1_c0；轨迹是 T_wc，所以累乘相对位姿的逆矩阵。
            T_c1_c0 = make_T(cv2.Rodrigues(rvec)[0], tvec)
            poses.append(poses[-1] @ np.linalg.inv(T_c1_c0))
            stamps.append(row[0])
            kp0, des0, depth0 = kp1, des1, depth1

        except (RuntimeError, cv2.error) as error:
            log["status"] = f"lost: {error}"
            logs.append(log)
            break

        logs.append(log)

    # 保存每一步的匹配数、PnP 内点和重投影误差。
    with (out / "tracking.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)

    write_tum(out / "trajectory_tum.txt", stamps, poses)
    np.savez_compressed(out / "trajectory.npz", timestamps=stamps, poses_wc=poses)
    report = {
        "requested_frames": len(rows),
        "successful_poses": len(poses),
        "coverage": len(poses) / len(rows),
        "stopped_on_failure": logs[-1]["status"] != "ok"
    }
    write_json(out / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()