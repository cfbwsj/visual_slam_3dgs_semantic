#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ORB="$ROOT/third_party/ORB_SLAM3"

# 让程序找到安装在项目 .local 目录下的 Pangolin 动态库。
export LD_LIBRARY_PATH="$ROOT/.local/lib:$ROOT/.local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

OUT="$ROOT/outputs/orbslam3/fr1_xyz"
mkdir -p "$OUT"
cd "$OUT"

# 参数依次为：词袋文件、相机配置、数据集目录、RGB-D 时间关联文件。
"$ORB/Examples/RGB-D/rgbd_tum" \
  "$ORB/Vocabulary/ORBvoc.txt" \
  "$ORB/Examples/RGB-D/TUM1.yaml" \
  "$ROOT/data/raw/tum/rgbd_dataset_freiburg1_xyz" \
  "$ROOT/data/processed/tum_fr1/associations.txt"