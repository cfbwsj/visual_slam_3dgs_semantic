import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path

from vslam.data.tum import read_index
'''这段代码先为每张 RGB 图寻找时间差在容差范围内的深度图候选，再按时间差
从小到大做全局贪心一对一匹配，最后按 RGB 时间排序并写出
rgb_time rgb_path depth_time depth_path 格式的关联文件。'''

def associate(rgb_rows, depth_rows, max_difference=0.02):
    # 先为每个 RGB 时间戳找出容差范围内的深度候选。
    depth_times = [row[0] for row in depth_rows]
    candidates = []

    for rgb_id, (rgb_time, _) in enumerate(rgb_rows):
        # bisect_left/right 在已排序时间戳中快速找到候选区间。
        left = bisect_left(depth_times, rgb_time - max_difference)
        right = bisect_right(depth_times, rgb_time + max_difference)

        for depth_id in range(left, right):
            gap = abs(rgb_time - depth_times[depth_id])
            candidates.append((gap, rgb_id, depth_id))

    # 按时间差从小到大匹配，并保证每张 RGB / 深度图只使用一次。
    candidates.sort()
    used_rgb, used_depth, selected = set(), set(), []

    for gap, rgb_id, depth_id in candidates:
        if rgb_id in used_rgb or depth_id in used_depth:
            continue

        selected.append((*rgb_rows[rgb_id], *depth_rows[depth_id]))
        used_rgb.add(rgb_id)
        used_depth.add(depth_id)

    # 输出顺序按 RGB 时间戳排列。
    return sorted(selected, key=lambda row: row[0])


def read_associations(path):
    # 读取关联文件，每行格式为：rgb_time rgb_path depth_time depth_path。
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rgb_time, rgb_path, depth_time, depth_path = line.split()
        rows.append((float(rgb_time), rgb_path, float(depth_time), depth_path))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-difference", type=float, default=0.02)
    args = parser.parse_args()

    root = Path(args.root)
    rgb_rows = read_index(root / "rgb.txt")
    depth_rows = read_index(root / "depth.txt")
    rows = associate(rgb_rows, depth_rows, args.max_difference)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rgb_time, rgb_path, depth_time, depth_path in rows:
            f.write(
                f"{rgb_time:.6f} {rgb_path} "
                f"{depth_time:.6f} {depth_path}\n"
            )

    print(f"已生成 {len(rows)} 对 RGB-D 关联")

if __name__ == "__main__":
    main()