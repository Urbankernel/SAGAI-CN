"""全局配置：路径、密钥、服务端点。

所有密钥走环境变量（可参考 .env.example），不落库、不进代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv 为可选依赖
    pass


def _split_keys(raw: str) -> list[str]:
    return [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]


@dataclass
class Config:
    """运行配置。默认值即可用，全部可被环境变量覆盖。"""

    # ---- 存储（替代 Google Drive）----
    base_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SAGAI_BASE", "./data"))
    )

    # ---- 百度全景（替代 Google Street View）----
    # 支持多个 AK 轮换，英文逗号/分号分隔
    baidu_aks: list[str] = field(
        default_factory=lambda: _split_keys(os.environ.get("BAIDU_AK", ""))
    )
    baidu_pano_url: str = "https://api.map.baidu.com/panorama/v2"

    # ---- 天地图底图（替代 CARTO / OSM 瓦片）----
    tianditu_tk: str = field(default_factory=lambda: os.environ.get("TIANDITU_TK", ""))

    # ---- Overpass（可切国内可达镜像；或用本地数据替代，见 sampling.py）----
    overpass_url: str = field(
        default_factory=lambda: os.environ.get(
            "OVERPASS_URL", "https://overpass-api.de/api/interpreter"
        )
    )

    # ---- 模型 ----
    model_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SAGAI_MODELS", "./models"))
    )
    dashscope_api_key: str = field(
        default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", "")
    )
    dashscope_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )

    # ---- 目录约定（与上游 SAGAI 的 Drive 目录结构对应）----
    @property
    def samples_dir(self) -> Path:
        return self.base_dir / "StreetSamples"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "Spatial_Results"

    def images_dir(self) -> Path:
        return self.base_dir / "StreetViewBatchDownload"

    def gpkg_path(self) -> Path:
        return self.samples_dir / "street_samples.gpkg"

    def ensure_dirs(self) -> None:
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
