import cv2
import numpy as np

from vslam.common import make_T, project, transform


def estimate_pair(p0, p1, K, distortion, ransac_px=1.5,
                  reprojection_px=3.0, min_parallax_deg=0.5):
    if len(p0) < 8 or len(p0) != len(p1):
        raise ValueError("对应点不足或数量不一致")

    # undistortPoints 去除镜头畸变，输出归一化相机坐标。
    n0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K, distortion).reshape(-1, 2)
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, distortion).reshape(-1, 2)

    # findEssentialMat 用 RANSAC 估计本质矩阵 E，并标记几何内点。
    threshold = ransac_px / ((K[0, 0] + K[1, 1]) / 2)
    E, mask = cv2.findEssentialMat(
        n0, n1, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=threshold
    )
    if E is None or mask is None:
        raise RuntimeError("本质矩阵估计失败")

    # E 可能包含多个 3×3 候选；recoverPose 用正深度约束选出可行位姿。
    solutions = [
        cv2.recoverPose(Ei, n0, n1, np.eye(3), mask=mask.copy())
        for Ei in np.split(E, E.shape[0] // 3)
    ]
    count, R, t, pose_mask = max(solutions, key=lambda item: item[0])
    ids = np.flatnonzero(pose_mask.ravel())

    # triangulatePoints 输入两个投影矩阵和对应归一化坐标，输出齐次三维点。
    P0 = np.column_stack([np.eye(3), np.zeros(3)])
    P1 = np.column_stack([R, t])
    Xh = cv2.triangulatePoints(P0, P1, n0[ids].T, n1[ids].T).T
    finite = np.abs(Xh[:, 3]) > 1e-10
    X = Xh[:, :3] / Xh[:, 3:4]
    X1 = transform(make_T(R, t), X)

    # 将归一化坐标乘 K 转回像素坐标，计算两帧重投影误差。
    uv0 = project(K, np.column_stack([n0[ids], np.ones(len(ids))]))
    uv1 = project(K, np.column_stack([n1[ids], np.ones(len(ids))]))
    e0 = np.linalg.norm(project(K, X) - uv0, axis=1)
    e1 = np.linalg.norm(project(K, X1) - uv1, axis=1)

    # 两条视线夹角越小，三角化深度越不稳定。
    C1 = (-R.T @ t).ravel()
    ray0, ray1 = X, X - C1
    denominator = np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1)
    cosine = np.sum(ray0 * ray1, axis=1) / denominator
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    # 保留正深度、有限坐标、低重投影误差和足够视差的点。
    valid = (finite & np.isfinite(X).all(axis=1) & (X[:, 2] > 0)
             & (X1[:, 2] > 0) & (e0 < reprojection_px)
             & (e1 < reprojection_px) & (angles >= min_parallax_deg))
    if valid.sum() < 8:
        raise RuntimeError("过滤后有效三维点不足")

    report = {
        "matches": len(p0),
        "ransac_inliers": int(np.count_nonzero(mask)),
        "recover_pose_inliers": int(count),
        "valid_points": int(valid.sum()),
        "median_reprojection_px": float(np.median(np.maximum(e0, e1)[valid])),
        "median_parallax_deg": float(np.median(angles[valid])),
        "scale": "arbitrary; baseline translation has unit norm"
    }
    result = dict(
        T_c1_c0=make_T(R, t), points_c0=X[valid],
        uv0=uv0[valid], uv1=uv1[valid], K=K, match_indices=ids[valid]
    )
    return result, report