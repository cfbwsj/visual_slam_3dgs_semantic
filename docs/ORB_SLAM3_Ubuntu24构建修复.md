# ORB-SLAM3 在本机的构建修复

本记录对应 Ubuntu 24.04、GCC 13.3、CMake 3.28.3、C++ OpenCV 4.6.0。

- Pangolin v0.6 commit：`dd801d244db3a8e27b7fe8020cd751404aa818fd`
- ORB-SLAM3 commit：`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`

## 已定位的问题与处理

1. `image_io_jpg.cpp` 使用 `uint8_t` 却未直接包含 `<cstdint>`。保留此前已添加的头文件。
2. `include/pangolin/log/packetstream_tags.h` 使用 `uint32_t` 却未直接包含 `<cstdint>`。补充头文件，消除日志模块的连带类型错误。
3. Pangolin v0.6 的 FFmpeg 驱动使用 `av_register_all`、`AVStream::codec` 等本机 FFmpeg 已不再提供的接口。构建中设置 `BUILD_PANGOLIN_FFMPEG=OFF`，保留 Pangolin GUI。TUM RGB-D 图像通过 OpenCV 读取，因此不依赖这个视频驱动。

`scripts/build_orbslam3.sh` 另外关闭依赖的示例/工具/测试程序构建，并指定 `rgbd_tum` 目标。没有关闭编译器错误检查，也没有改变 SLAM 算法。`BUILD_JOBS` 可控制编译并发，默认 2。

## 增量重建

在项目根目录运行：

```bash
set -o pipefail
bash scripts/build_orbslam3.sh 2>&1 | tee outputs/build/orbslam3.log
```

如果内存紧张：

```bash
BUILD_JOBS=1 bash scripts/build_orbslam3.sh
```

请保留 `build` 目录，以复用已经成功的编译结果。

## 新机器复现头文件补丁

两个头文件修改已保存为 `docs/patches/pangolin-v0.6-gcc13.patch`。仅在重新克隆、尚未修改的上述 Pangolin 版本上执行：

```bash
git -C third_party/Pangolin apply --check ../../docs/patches/pangolin-v0.6-gcc13.patch
git -C third_party/Pangolin apply ../../docs/patches/pangolin-v0.6-gcc13.patch
```

当前工作目录已经包含补丁，不要重复应用。第三方仓库通常被根项目忽略，因此应保留此补丁和构建脚本。

## 日志

本次修复已验证：

- `bash scripts/build_orbslam3.sh` 返回 0，日志末尾为 `[100%] Built target rgbd_tum`。
- `Examples/RGB-D/rgbd_tum` 已生成，主库 `lib/libORB_SLAM3.so` 链接成功。
- 按运行脚本设置库路径后，`ldd` 没有 `not found`。
- 无参数启动程序能打印正确用法，并返回预期的参数错误码 1。
- 本次验证未启动图形窗口，也未重新运行完整 TUM 序列；轨迹质量需在后续实验中评价。

本次修复的构建输出位于 `outputs/build/orbslam3_fix.log`，动态库检查和入口输出分别保存为 `rgbd_tum_ldd.log`、`rgbd_tum_usage.log`。最终运行仍使用：

```bash
bash scripts/run_orbslam3.sh
```

可视化运行需要可用的显示服务及 OpenGL 环境。
