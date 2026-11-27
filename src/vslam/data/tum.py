from pathlib import Path

import cv2
import numpy as np


def read_index(path):
    """读取 TUM 的 timestamp relative_path 索引。"""
    with Path(path).open("r", encoding="utf-8") as f:
        rows = [
            (float(ts), rel)
            for raw in f
            if (line := raw.strip()) and not line.startswith("#")
            for ts, rel in [line.split()]
        ]
    return rows


class TUMSequence:
    def __init__(self, root):
        self.root = Path(root)
        self.rgb_rows = read_index(self.root / "rgb.txt")
        self.depth_rows = read_index(self.root / "depth.txt")

    def __len__(self):
        return len(self.rgb_rows)

    def read_rgb(self, index):
        timestamp, relative_path = self.rgb_rows[index]
        image = cv2.imread(str(self.root / relative_path), cv2.IMREAD_COLOR)
        return timestamp, image  # BGR

    def read_depth(self, index):
        timestamp, relative_path = self.depth_rows[index]
        depth = cv2.imread(str(self.root / relative_path), cv2.IMREAD_UNCHANGED)
        return timestamp, depth  # uint16, 单位 mm