# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import argparse                          # 命令行参数
import json                              # 写报告
from pathlib import Path                 # 路径操作

import cv2                               # 读写图、画关键点/匹配、drawMatches
import numpy as np                       # K 矩阵、npz 存储

from vslam.config import load_config     # 项目内：读 YAML 配置
from vslam.data.tum import TUMSequence   # 项目内：TUM 数据集读取
from vslam.vo.features import (          # 项目内：VO 特征模块
    extract_orb,                         # 提 ORB 关键点 + 描述子
    match_ratio,                         # KNN + ratio test + 去重
    matched_points,                      # 匹配 → 像素坐标数组
)


def main():
    # 1) 命令行接口
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)    # YAML 配置路径
    parser.add_argument("--out", required=True)       # 输出目录
    parser.add_argument("--index0", type=int)         # 可选：覆盖 cfg 里的 i
    parser.add_argument("--index1", type=int)         # 可选：覆盖 cfg 里的 j
    args = parser.parse_args()

    # 2) 加载配置 + 打开 TUM 序列
    cfg = load_config(args.config)
    sequence = TUMSequence(cfg["dataset"]["root"])

    # 3) 确定两帧的索引：命令行 > 配置默认值
    #    args.index0 为 None 说明没传，回落到 cfg["pair"]["index0"]
    i = cfg["pair"]["index0"] if args.index0 is None else args.index0
    j = cfg["pair"]["index1"] if args.index1 is None else args.index1

    # 4) 准备输出目录（多级创建，已存在不报错）
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 5) 读两帧 RGB，拿到时间戳（后续对齐/评估要用）
    t0, image0 = sequence.read_rgb(i)                 # image: (H,W,3) uint8 BGR
    t1, image1 = sequence.read_rgb(j)

    # 6) 从配置取 ORB 与匹配参数
    nfeatures = cfg["orb"]["nfeatures"]               # 每帧特征数上限
    ratio = cfg["matching"]["ratio"]                  # Lowe's ratio 阈值

    # 7) 两帧分别提取 ORB
    keypoints0, descriptors0 = extract_orb(image0, nfeatures)
    keypoints1, descriptors1 = extract_orb(image1, nfeatures)

    # 8) 匹配 + 把匹配索引翻译成像素坐标
    matches = match_ratio(descriptors0, descriptors1, ratio)
    points0, points1 = matched_points(keypoints0, keypoints1, matches)

    # 9) 可视化：图 0 的关键点（DRAW_RICH_KEYPOINTS 会画出方向/尺度）
    keypoint_view = cv2.drawKeypoints(
        image0, keypoints0, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # 10) 可视化：两张图并排 + 连线（只画前 100 条，避免图太花）
    #     画的是 matches 的前 100 条，因为已经按 distance 升序，前 100 条质量最高
    match_view = cv2.drawMatches(
        image0, keypoints0, image1, keypoints1, matches[:100], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # 未匹配的点不画
    )

    # 11) 存两张可视化图
    cv2.imwrite(str(out / "keypoints.png"), keypoint_view)
    cv2.imwrite(str(out / "matches.png"), match_view)

    # 12) 构造相机内参矩阵 K（3×3）
    #     针孔模型：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera = cfg["camera"]
    K = np.array([
        [camera["fx"], 0,             camera["cx"]],
        [0,            camera["fy"],  camera["cy"]],
        [0,            0,             1],
    ], dtype=np.float64)

    # 13) 把这一对帧的全部数据打包成 npz，供后续几何验证/三角化使用
    #     显式指定 dtype 是必要的：savez 对 Python float/int 的默认转换不如显式清晰
    #     np.savez_compressed 压缩存储，points 阵列通常不大但压缩后体积更友好
    np.savez_compressed(
        out / "pair.npz",
        points0=points0,                              # (N,2) float64，图 0 的像素坐标
        points1=points1,                              # (N,2) float64，图 1 的像素坐标
        timestamp0=np.float64(t0),                    # 图 0 时间戳（秒）
        timestamp1=np.float64(t1),                    # 图 1 时间戳（秒）
        index0=np.int64(i),                           # 帧索引，方便溯源
        index1=np.int64(j),
        K=K,                                          # 3×3 内参
        distortion=np.asarray(camera["distortion"], dtype=np.float64),  # 5 个畸变系数
    )

    # 14) 组装统计报告
    report = {
        "index0": i,
        "index1": j,
        "keypoints0": len(keypoints0),                # 图 0 检测到的特征数
        "keypoints1": len(keypoints1),                # 图 1 检测到的特征数
        "matches_after_ratio": len(matches),          # ratio test + 去重后的匹配数
        "ratio": ratio,
        "nfeatures": nfeatures,
        "geometry_verified": False,                   # 本脚本不做几何验证，留待后续
    }

    # 15) 写报告 + 打印（两份内容一致，一份存档一份即时可见）
    (out / "match_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))

    # 16) 业务门槛：8 点法需要至少 8 对匹配才能解基础矩阵
    #     这不是防御，是功能约束 —— 低于 8 对继续下去没有几何意义
    if len(matches) < 8:
        raise RuntimeError("匹配太少，先检查图像和参数，不进入位姿估计")


if __name__ == "__main__":
    main()