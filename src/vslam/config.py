from pathlib import Path

import yaml


def load_config(path):
    config_path = Path(path).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 约定配置位于项目根目录下的 configs 文件夹
    project_root = config_path.parent.parent
    dataset_root = Path(cfg["dataset"]["root"]).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    cfg["dataset"]["root"] = str(dataset_root.resolve())
    return cfg