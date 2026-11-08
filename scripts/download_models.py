#!/usr/bin/env python
"""预下载 ModelScope 模型权重（等同 cli download-models）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sagai_cn.config import Config
from sagai_cn.models import MODEL_REGISTRY, download_model

if __name__ == "__main__":
    cfg = Config()
    names = sys.argv[1:] or list(MODEL_REGISTRY)
    for n in names:
        print(f"== {n} ==")
        download_model(n, cfg.model_dir)
    print("全部完成。")
