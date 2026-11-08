"""Block 3 国产化：模型获取与加载。

路线 A（默认）：ModelScope 下载权重 -> 本地路径加载。
  Qwen2.5-VL / Qwen3-VL / InternVL3.5 均为国产模型，ModelScope 有官方镜像，
  无需 HF token，国内满速下载。LLaVA-NeXT / Gemma 无国内官方镜像，不内置。
路线 B：无 GPU 时走 DashScope 云 API（见 inference.DashScopeEngine）。
路线 C：纯 CPU 走 GGUF + llama.cpp（见 inference.CpuEngine 与 GGUF_REGISTRY）。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

# 内置可选模型（均为 ModelScope 官方仓库 id，也是 HF 同名仓库）
MODEL_REGISTRY: dict[str, dict] = {
    "Qwen3-VL-8B-Instruct": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "vram_4bit_gb": 7,
        "note": "推荐默认：效果好，单卡 8GB+ 可跑 4bit",
    },
    "Qwen3-VL-4B-Instruct": {
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "vram_4bit_gb": 4,
        "note": "轻量，6GB 显存可跑",
    },
    "Qwen2.5-VL-7B-Instruct": {
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "vram_4bit_gb": 6,
        "note": "上游 SAGAI 默认模型，便于与原版结果对照",
    },
    "Qwen2.5-VL-32B-Instruct": {
        "repo": "Qwen/Qwen2.5-VL-32B-Instruct",
        "vram_4bit_gb": 20,
        "note": "高精度选项，需 24GB 级显卡",
    },
    "InternVL3_5-8B": {
        "repo": "OpenGVLab/InternVL3_5-8B",
        "vram_4bit_gb": 7,
        "note": "上海 AI Lab，多模态基准强",
    },
}

# GGUF 量化模型（路线 C：纯 CPU 打分，llama.cpp / llama-server）
# 每个模型 = 语言权重 LLM + 视觉投影 mmproj 两个文件，必须配套下载。
GGUF_REGISTRY: dict[str, dict] = {
    "Qwen3-VL-8B-Instruct-GGUF": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "llm": "Qwen3VL-8B-Instruct-{quant}.gguf",
        "mmproj": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "ram_gb": "Q4_K_M≈6.5 / Q8_0≈8.7",
        "note": "官方仓库，纯 CPU 推荐 q4_k_m",
    },
}

# 量化档位别名：命令行小写 -> GGUF 文件名大写后缀
QUANT_ALIASES = {
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q8_0": "Q8_0",
    "f16": "F16",
}

# GGUF 下载源 URL 模板（{repo}、{filename} 为占位符）。
# 默认用 ModelScope：Qwen 官方 GGUF 仓库在 HF 上走 Xet 存储，hf-mirror 会把
# 请求重定向到海外 AWS CDN（us.aws.cdn.hf.co），国内 SSL 连接被重置导致下载失败；
# ModelScope 同名仓库文件名一致，走国内 CDN 直连满速。
GGUF_SOURCES = {
    "modelscope": "https://modelscope.cn/models/{repo}/resolve/master/{filename}",
    "hf-mirror": "https://hf-mirror.com/{repo}/resolve/main/{filename}",
    "hf": "https://huggingface.co/{repo}/resolve/main/{filename}",
}


def download_model(name: str, model_dir: Path) -> Path:
    """从 ModelScope 下载模型权重到本地。返回本地路径。"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}，可选：{list(MODEL_REGISTRY)}")
    repo = MODEL_REGISTRY[name]["repo"]
    local = model_dir / name
    if (local / "config.json").exists():
        print(f"已存在，跳过下载：{local}")
        return local
    local.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope import snapshot_download
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "modelscope",
             "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
        )
        from modelscope import snapshot_download
    print(f"从 ModelScope 下载 {repo} -> {local}（首次较慢，请耐心等待）")
    snapshot_download(repo, local_dir=str(local))
    return local


def download_gguf(name: str, model_dir: Path, quant: str = "q4_k_m",
                  source: str = "modelscope") -> dict[str, Path]:
    """路线 C：下载 GGUF 模型（LLM 权重 + mmproj 视觉投影，双文件配套）。

    默认走 ModelScope（魔搭，国内直连满速）。Qwen 官方 GGUF 仓库在 HF 上
    用 Xet 存储，hf-mirror 会重定向到海外 AWS CDN 导致 SSL 失败，故默认改
    用 ModelScope。``source`` ∈ modelscope / hf-mirror / hf。
    返回 {"llm": Path, "mmproj": Path}，文件落 model_dir/gguf/。
    """
    if name not in GGUF_REGISTRY:
        raise KeyError(f"未知 GGUF 模型 {name}，可选：{list(GGUF_REGISTRY)}")
    if source not in GGUF_SOURCES:
        raise ValueError(f"未知下载源 {source}，可选：{list(GGUF_SOURCES)}")
    spec = GGUF_REGISTRY[name]
    repo = spec["repo"]
    qname = QUANT_ALIASES.get(quant, quant.upper())
    llm_name = spec["llm"].format(quant=qname)
    mmproj_name = spec["mmproj"]

    out_dir = model_dir / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, filename in (("llm", llm_name), ("mmproj", mmproj_name)):
        dest = out_dir / filename
        paths[label] = dest
        if dest.exists() and dest.stat().st_size > 0:
            print(f"已存在，跳过下载：{dest}")
            continue
        url = GGUF_SOURCES[source].format(repo=repo, filename=filename)
        print(f"下载 {filename}")
        print(f"  {url}（首次较慢，请耐心等待）")
        _download_file(url, dest)
    return paths


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20,
                   retries: int = 3) -> None:
    """流式下载，带 .part 断点续传与进度显示。404 时给仓库核对提示。

    SSL 抖动 / 连接中断（国内访问海外 CDN 常见）会自动退避重试，且每次都
    从已下完的 .part 断点继续，不从头再来。
    """
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 416:  # 服务器认为已完整
                    part.replace(dest)
                    print(f"  已完成：{dest.name}")
                    return
                if r.status_code == 404:
                    raise FileNotFoundError(
                        f"文件不存在（404）：{url}\n"
                        f"请打开仓库核对实际文件名，或改 GGUF_REGISTRY 里的文件名")
                r.raise_for_status()
                total = resume_from + (int(r.headers.get("content-length", 0) or 0))
                mode = "ab" if resume_from else "wb"
                done = resume_from
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 // total
                            print(f"\r  {done // (1 << 20)}/{total // (1 << 20)} MB "
                                  f"({pct}%)", end="", flush=True)
                print()
            part.replace(dest)
            print(f"  完成：{dest.name}")
            return
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            if attempt == retries - 1:
                break
            wait = 2 ** attempt
            print(f"\n  连接中断（第 {attempt + 1}/{retries} 次），"
                  f"{wait}s 后从断点重试…")
            time.sleep(wait)
    raise ConnectionError(
        f"下载失败（重试 {retries} 次后仍失败）：{url}\n"
        f"  原因：{last_err}\n"
        f"  提示：换下载源重试，如 --gguf-source modelscope / hf；"
        f"或手动下载后把文件放到 {part.parent} 再重跑") from last_err


def _infer_uvlm_backend(model_path: Path, model_choices: dict) -> str:
    """按本地模型目录名推断 UVLM 的 backend（qwen3/qwen/internvl/gemma4/llava）。"""
    name = model_path.name
    # 1) 目录名 == 注册表 repo_id 末段（如 Qwen3-VL-8B-Instruct）时直接命中
    for _key, (_backend, repo_id) in model_choices.items():
        if repo_id.rstrip("/").split("/")[-1] == name:
            return _backend
    # 2) 兜底：按名字关键字
    low = name.lower()
    if "qwen3" in low:
        return "qwen3"
    if "qwen2" in low or "qwen-vl" in low:
        return "qwen"
    if "internvl" in low:
        return "internvl"
    if "gemma" in low:
        return "gemma4"
    if "llava" in low:
        return "llava"
    raise ValueError(
        f"无法从模型目录名 {name!r} 推断 UVLM 后端类型，"
        f"请确认模型名含 qwen3/qwen2/internvl/gemma/llava 关键字")


def load_model_uvlm(model_path: Path, precision: str = "4bit",
                    device_map: str = "auto"):
    """路线 A：通过 UVLM 加载本地权重（保持与上游一致的推理栈）。

    UVLM 的 load_model 靠 MODEL_CHOICES 表按模型名查 (backend, HF repo_id)，
    再 from_pretrained(repo_id)——默认会去 HF 在线下载。国产化后权重已由
    ModelScope 下到本地，故把本地路径注册进该表（key 与 model_id 都用本地
    路径），backend 按目录名推断，从而复用 UVLM 的加载/推理逻辑且不联网。
    """
    try:
        from uvlm import load_model
        from uvlm.registry import MODEL_CHOICES
    except ImportError as e:
        raise ImportError(
            "本地推理需要 UVLM（上游推理引擎）。安装：\n"
            "  pip install ./vendor/UVLM-4.0.0.zip  （离线包，见 README）\n"
            "或无 GPU 时改用路线 B：cli infer --engine dashscope"
        ) from e

    backend = _infer_uvlm_backend(model_path, MODEL_CHOICES)
    key = str(model_path)
    # 注册本地路径：model_id 即本地路径，from_pretrained 直接读，不再连 HF
    MODEL_CHOICES[key] = (backend, key)
    return load_model(
        model_name=key,
        precision=precision,
        device_map=device_map,
        low_cpu_mem_usage=True,
        hf_token=None,  # 国产化后无需 HF token
    )
