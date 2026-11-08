"""Block 5 国产化：批量推理引擎。

两个引擎：
  LocalEngine    —— 路线 A，委托 UVLM run_batch（本地 GPU，功能与上游完全一致）
  DashScopeEngine —— 路线 B，阿里云百炼 OpenAI 兼容接口（无 GPU 备选），
                    自实现共识投票 / CoT / 截断检测 / 断点续跑

CSV 契约（与上游一致）：每行一张影像；image_name + 每任务一列分数，
另附 {col}_raw 原始输出、{col}_reasoning 思维链（如有）、{col}_truncated 标记。
"""
from __future__ import annotations

import base64
import csv
import re
import time
from pathlib import Path

import requests

from .tasks import TaskSpec, GenerationConfig

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _ensure_utf8_sig(csv_path: Path) -> None:
    """给 CSV 补 UTF-8 BOM（若无），Excel 双击打开中文不乱码。

    UVLM 的 batch.py 用默认 utf-8（无 BOM）写 CSV，Excel 会按系统编码猜导致
    中文乱码；DashScope/Cpu 引擎则用 utf-8-sig。这里做字节级补写：不改内容、
    幂等，且不影响 pandas 后续读取（实测 utf-8 默认编码读 utf-8-sig 列名正常）。
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    data = csv_path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return
    csv_path.write_bytes(b"\xef\xbb\xbf" + data)


# ------------------------------------------------------------------
# 答案解析（与上游 UVLM 行为对齐：优先 ANSWER: 行，兜底正则）
# ------------------------------------------------------------------
def parse_answer(raw: str, task_type: str):
    """从模型输出解析最终答案。无法解析返回 "NA"。"""
    if not raw:
        return "NA"
    m = re.findall(r"ANSWER:\s*(.+)", raw, flags=re.IGNORECASE)
    tail = m[-1].strip() if m else raw.strip().splitlines()[-1].strip()

    if task_type == "numeric":
        nums = re.findall(r"-?\d+(?:\.\d+)?", tail if m else raw)
        return nums[-1] if nums else "NA"
    if task_type == "boolean":
        low = tail.lower()
        # 先判否定（"没有"包含"有"，顺序不能反）；中文关键词不用 \b（CJK 间无词边界）
        if re.search(r"\bno\b", low) or any(k in tail for k in ("没有", "不是", "否", "无")):
            return "no"
        if re.search(r"\byes\b", low) or any(k in tail for k in ("是", "有")):
            return "yes"
        return "NA"
    if task_type == "category":
        return tail.split()[0].strip("。.,，") if tail else "NA"
    return raw.strip()  # text 原样


def _consensus(values: list[str], task_type: str, tolerance_pct: float) -> str:
    """多数投票；numeric 在容差内取中位数。无共识返回 "NA"。"""
    valid = [v for v in values if v != "NA"]
    if not valid:
        return "NA"
    if task_type == "numeric":
        nums = sorted(float(v) for v in valid)
        median = nums[len(nums) // 2]
        tol = abs(median) * tolerance_pct / 100 if tolerance_pct else 0.0
        agreed = [n for n in nums if abs(n - median) <= max(tol, 1e-9)]
        if len(agreed) > len(values) / 2:
            return str(int(median) if median.is_integer() else median)
        return "NA"
    # category / boolean：简单多数
    for v in set(valid):
        if valid.count(v) > len(values) / 2:
            return v
    return "NA"


# ------------------------------------------------------------------
# 影像枚举与断点续跑
# ------------------------------------------------------------------
def list_images(image_folder: Path) -> list[Path]:
    """有效影像清单：排除 _NA 空图/重复图（与上游 Block 6 口径一致）。"""
    files = [p for p in sorted(image_folder.iterdir())
             if p.suffix.lower() in IMAGE_EXTS]
    return [p for p in files if "_NA" not in p.stem]


def _load_done_images(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    # utf-8-sig：兼容带/不带 BOM 的旧文件
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return {row["image_name"] for row in csv.DictReader(f) if row.get("image_name")}


def _csv_fieldnames(tasks: list[TaskSpec]) -> list[str]:
    names = ["image_name"]
    for t in tasks:
        names.append(t.column)
        names.append(f"{t.column}_raw")
        if t.advanced_reasoning:
            names.append(f"{t.column}_reasoning")
        names.append(f"{t.column}_truncated")
    return names


# ------------------------------------------------------------------
# 路线 A：本地 UVLM 引擎
# ------------------------------------------------------------------
class LocalEngine:
    """委托 UVLM 的本地推理（需 GPU）。行为与上游 Block 5 完全一致。"""

    name = "local"

    def __init__(self, model_ctx):
        self.model_ctx = model_ctx

    def run_batch(self, tasks: list[TaskSpec], image_folder: Path,
                  output_csv: Path, gen: GenerationConfig):
        from uvlm.batch import run_batch  # 延迟导入，无 GPU 环境也能 import 本模块

        result = run_batch(
            model_ctx=self.model_ctx,
            task_specs=[t.to_dict() for t in tasks],
            image_folder=str(image_folder),
            output_path=str(output_csv),
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
            seed=gen.seed,
        )
        # UVLM 写 CSV 无 BOM，Excel 双击中文乱码；补 BOM 与云/CPU 路线对齐
        _ensure_utf8_sig(output_csv)
        return result


# ------------------------------------------------------------------
# 路线 B：DashScope 云 API 引擎（OpenAI 兼容）
# ------------------------------------------------------------------
class DashScopeEngine:
    """阿里云百炼 qwen-vl 系列。无 GPU 备选；按 token 计费。"""

    name = "dashscope"

    def __init__(self, api_key: str, model: str = "qwen3-vl-plus",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 rpm: int = 60, max_retries: int = 4):
        if not api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.interval = 60.0 / max(rpm, 1)
        self.max_retries = max_retries
        self._last_ts = 0.0

    def _chat(self, image_path: Path, prompt: str, gen: GenerationConfig) -> dict:
        """单次视觉问答。返回 {text, truncated}。"""
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": gen.max_new_tokens,
            "temperature": gen.temperature if gen.do_sample else 0.0,
            "top_p": gen.top_p,
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            wait = self.interval - (time.time() - self._last_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.time()
            try:
                r = requests.post(self.base_url, json=payload,
                                  headers=headers, timeout=120)
                if r.status_code == 429:  # 限流：指数退避
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                r.raise_for_status()
                choice = r.json()["choices"][0]
                return {
                    "text": choice["message"]["content"],
                    "truncated": choice.get("finish_reason") == "length",
                }
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"DashScope 调用失败（{image_path.name}）：{last_err}")

    def _run_task(self, image_path: Path, task: TaskSpec,
                  gen: GenerationConfig, role: str) -> dict:
        prompt = task.build_prompt(role)
        runs = task.consensus_runs if task.consensus_enabled else 1
        raws, answers, truncated_any = [], [], False
        reasoning = ""
        for _ in range(runs):
            out = self._chat(image_path, prompt, gen)
            raws.append(out["text"])
            truncated_any = truncated_any or out["truncated"]
            answers.append(parse_answer(out["text"], task.task_type))
            if task.advanced_reasoning and not reasoning:
                reasoning = out["text"]
        value = (_consensus(answers, task.task_type, task.numeric_tolerance)
                 if task.consensus_enabled else answers[0])
        return {"value": value, "raw": raws[0] if runs == 1 else " ||| ".join(raws),
                "reasoning": reasoning, "truncated": truncated_any}

    def run_batch(self, tasks: list[TaskSpec], image_folder: Path,
                  output_csv: Path, gen: GenerationConfig, role: str = ""):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _csv_fieldnames(tasks)
        done = _load_done_images(output_csv)
        images = [p for p in list_images(image_folder) if p.name not in done]
        print(f"待处理 {len(images)} 张（已完成 {len(done)} 张，断点续跑）")

        write_header = not output_csv.exists()
        # utf-8-sig：带 BOM，Excel 双击打开中文不乱码
        with open(output_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for i, img in enumerate(images, 1):
                row = {"image_name": img.name}
                for task in tasks:
                    try:
                        res = self._run_task(img, task, gen, role)
                    except Exception as e:
                        print(f"  [warn] {img.name}/{task.column}: {e}")
                        res = {"value": "NA", "raw": "", "reasoning": "",
                               "truncated": False}
                    row[task.column] = res["value"]
                    row[f"{task.column}_raw"] = res["raw"]
                    if task.advanced_reasoning:
                        row[f"{task.column}_reasoning"] = res["reasoning"]
                    row[f"{task.column}_truncated"] = int(res["truncated"])
                writer.writerow(row)
                f.flush()  # 逐行落盘，中断不丢进度
                if i % 20 == 0 or i == len(images):
                    print(f"  进度 {i}/{len(images)}")
        print(f"完成：{output_csv}")


# ------------------------------------------------------------------
# 路线 C：本地 llama.cpp 引擎（GGUF 量化，纯 CPU 可用）
# ------------------------------------------------------------------
class CpuEngine(DashScopeEngine):
    """本地 llama-server 子进程 + OpenAI 兼容 HTTP（纯 CPU 打分）。

    与 DashScopeEngine 同构：复用其 _run_task / run_batch（共识投票 /
    CoT / 断点续跑 / CSV 契约），仅替换 _chat——把请求发往本地
    llama-server（默认 127.0.0.1:8080）而非阿里云。使用前按
    scripts/setup_cpu_engine.md 下载 GGUF 并启动 llama-server。
    """

    name = "cpu"

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1/chat/completions",
                 model: str = "qwen3-vl", max_retries: int = 4, timeout: int = 600):
        # 不调 super().__init__：本地服务无需 api_key / rpm 限流
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    def _chat(self, image_path: Path, prompt: str, gen: GenerationConfig) -> dict:
        """单次视觉问答，发往本地 llama-server。返回 {text, truncated}。"""
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": gen.max_new_tokens,
            "temperature": gen.temperature if gen.do_sample else 0.0,
            "top_p": gen.top_p,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.base_url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                return {
                    "text": choice["message"]["content"],
                    "truncated": choice.get("finish_reason") == "length",
                }
            except requests.exceptions.ConnectionError as e:
                last_err = e
                if attempt == 0:
                    print("无法连接 llama-server，请先按 scripts/setup_cpu_engine.md 启动服务")
                time.sleep(min(2 ** attempt, 8))
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"llama-server 调用失败（{image_path.name}）：{last_err}")
