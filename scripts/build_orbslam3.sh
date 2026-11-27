#!/usr/bin/env bash
set -euo pipefail

# 从脚本所在位置推导项目根目录，避免依赖当前终端目录。
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX="$ROOT/.local"
JOBS=${BUILD_JOBS:-2}
export CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# 先编译并安装 Pangolin，ORB-SLAM3 的可视化窗口会用到它。
cmake -S "$ROOT/third_party/Pangolin" -B "$ROOT/third_party/Pangolin/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_PANGOLIN_PYTHON=OFF \
  -DBUILD_PANGOLIN_FFMPEG=OFF \
  -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF
# TUM 图像由 OpenCV 读取，无需 Pangolin 的旧 FFmpeg 视频接口。
cmake --build "$ROOT/third_party/Pangolin/build" -j"$JOBS"
cmake --install "$ROOT/third_party/Pangolin/build"

# 编译 ORB-SLAM3 仓库自带的 DBoW2、g2o 和 Sophus。
ORB="$ROOT/third_party/ORB_SLAM3"
for library in DBoW2 g2o Sophus; do
  EXTRA_OPTIONS=()
  if [[ "$library" == Sophus ]]; then
    EXTRA_OPTIONS=(-DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF)
  fi
  cmake -S "$ORB/Thirdparty/$library" -B "$ORB/Thirdparty/$library/build" \
    -DCMAKE_BUILD_TYPE=Release "${EXTRA_OPTIONS[@]}"
  cmake --build "$ORB/Thirdparty/$library/build" -j"$JOBS"
done

# 首次运行时解压 ORB 词袋文件；后续运行不需要重复解压。
if [[ ! -f "$ORB/Vocabulary/ORBvoc.txt" ]]; then
  tar -xf "$ORB/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB/Vocabulary"
fi

# 最后编译 ORB-SLAM3 主程序和 RGB-D 示例。
cmake -S "$ORB" -B "$ORB/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ORB/build" --target rgbd_tum -j"$JOBS"
