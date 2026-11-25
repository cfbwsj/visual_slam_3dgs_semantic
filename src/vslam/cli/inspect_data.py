# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import argparse                         # 命令行参数解析（--config / --out）
import json                             # 输出结构化报告
from pathlib import Path                # 跨平台路径操作

import cv2                              # 图像读写、伪彩色映射
import numpy as np                      # 数组运算、掩码、中位数

from vslam.config import load_config    # 项目内：读 YAML 配置
from vslam.data.tum import TUMSequence  # 项目内：TUM 数据集读取封装


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    # 1) 定义命令行接口
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)   # 配置文件路径，必填
    parser.add_argument("--out", required=True)      # 输出目录，必填
    args = parser.parse_args()

    # 2) 读配置 + 打开 TUM 序列
    cfg = load_config(args.config)                   # 返回 dict，含 dataset/camera 等段
    sequence = TUMSequence(cfg["dataset"]["root"])   # root 指向 rgb.txt/depth.txt 所在目录

    # 3) 准备输出目录
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)           # 多级目录一起建；已存在不报错

    # 4) 读首帧 RGB 与深度（注意：这里分别取各自索引 0，尚未做时间对齐）
    timestamp_rgb, image = sequence.read_rgb(0)      # image: (H, W, 3) uint8, BGR
    timestamp_depth, depth = sequence.read_depth(0)  # depth: (H, W) uint16, 单位 mm

    # 5) 有效深度掩码
    #    TUM 深度图中 0 表示"无有效测量"（遮挡、超量程、反光等），
    #    后续中位数统计和可视化黑屏都依赖这个掩码
    valid = depth > 0

    # 6) 深度单位换算：uint16 毫米 → float32 米
    #    depth_scale=5000 时，像素值 5000 对应 1 米
    #    先 astype(float32) 再除，避免 uint16 直接除产生整数截断
    depth_m = depth.astype(np.float32) / cfg["camera"]["depth_scale"]

    # 7) 组装数据报告（纯 Python 原生类型，方便 json.dumps）
    report = {
        "rgb_frames": len(sequence.rgb_rows),          # RGB 索引行数
        "depth_frames": len(sequence.depth_rows),      # 深度索引行数（两者不一定相等）
        "first_rgb_timestamp": timestamp_rgb,          # 首帧 RGB 时间戳（秒，float）
        "first_depth_timestamp": timestamp_depth,      # 首帧深度时间戳（秒，float）
        "image_shape": list(image.shape),              # 转 list，否则 JSON 不能序列化 tuple
        "depth_dtype": str(depth.dtype),               # 如 "uint16"，str 化后才能 JSON 化
        "valid_depth_fraction": float(valid.mean()),   # 有效深度像素占比，bool 数组求均值即占比
        "median_depth_m": float(np.median(depth_m[valid])),  # 只对有效像素取中位数（米）
        "note": "此处分别检查首帧，尚未完成 RGB/深度时间关联",   # 明确当前脚本能力边界
    }

    # 8) 保存 RGB 原图（BGR 存储，PNG 里颜色通道与视觉一致，无需转换）
    cv2.imwrite(str(out / "first_rgb.png"), image)     # Path 转 str 给 OpenCV

    # 9) 生成深度伪彩色预览
    #    步骤 a：米 → [0,1] → [0,255]，截断到 5 米以内，超出的饱和
    #            clip 后乘 255 再转 uint8，作为伪彩色的输入索引图
    depth_view = np.uint8(np.clip(depth_m / 5.0, 0, 1) * 255)

    #    步骤 b：单通道索引图 → TURBO 伪彩色（近处暖色、远处冷色，视觉区分度高）
    depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_TURBO)

    #    步骤 c：无效深度（depth==0）区域涂黑，避免被伪彩色映射成误导性的颜色
    #            ~valid 是逻辑取反；True 的位置被赋值为 0（BGR 全黑）
    depth_view[~valid] = 0

    # 10) 保存深度预览图（注意：只是可视化，原始 depth 数组没被修改）
    cv2.imwrite(str(out / "first_depth.png"), depth_view)

    # 11) 把报告写入 JSON 文件
    #     ensure_ascii=False 让中文 note 直接可见而不是 \uXXXX
    #     indent=2 便于人工阅读
    (out / "data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 12) 同时打印到终端，方便快速查看
    print(json.dumps(report, indent=2, ensure_ascii=False))


# 脚本入口：直接 python xxx.py 时执行 main()
if __name__ == "__main__":
    main()