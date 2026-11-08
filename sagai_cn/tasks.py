"""Block 4 国产化：评分任务定义与 prompt 组装。

任务模型与上游 UVLM 的约定保持兼容（numeric/category/boolean/text、
CoT 时以 "ANSWER: <value>" 收尾），保证解析器行为一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

TASK_TYPES = ("numeric", "category", "boolean", "text")

# 与 UVLM ADVANCED_REASONING_FORMATS 对齐：CoT 任务要求最后一行给结论
ADVANCED_REASONING_FORMATS = {
    "numeric": "Think step by step, then end your response with a final line exactly like: ANSWER: <number>",
    "category": "Think step by step, then end your response with a final line exactly like: ANSWER: <category>",
    "boolean": "Think step by step, then end your response with a final line exactly like: ANSWER: yes 或 ANSWER: no",
}

DEFAULT_FORMATS = {
    "numeric": "Answer with only one integer number, nothing else.",
    "category": "Answer with only the category label, nothing else.",
    "boolean": "Answer with only yes or no, nothing else.",
    "text": "",
}


@dataclass
class TaskSpec:
    """单个评分任务。字段与上游 Block 4 的 task_specs 字典一一对应。"""

    column: str                      # CSV 列名，如 green_view
    prompt: str                      # 评分问题（中英文均可，国产模型中文更稳）
    task_type: str = "numeric"       # numeric/category/boolean/text
    theory: str = ""                 # 可选：定义、规则、边界情况
    fmt: str = ""                    # 输出格式说明；空则按 task_type 取默认
    advanced_reasoning: bool = False # CoT 思维链
    consensus_enabled: bool = False  # 共识投票
    consensus_runs: int = 3          # 2-5
    numeric_tolerance: float = 0.0   # numeric 共识容差（百分比）

    def __post_init__(self):
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type 必须是 {TASK_TYPES} 之一")
        if not 2 <= self.consensus_runs <= 5:
            raise ValueError("consensus_runs 须在 2-5 之间")
        if self.task_type == "text":
            # 与上游一致：text 任务不支持 CoT / 共识
            self.advanced_reasoning = False
            self.consensus_enabled = False

    def build_prompt(self, role: str = "") -> str:
        """组装完整 prompt：role + prompt + theory + format。"""
        fmt = self.fmt.strip()
        if self.advanced_reasoning:
            fmt = ADVANCED_REASONING_FORMATS.get(self.task_type, fmt)
        elif not fmt:
            fmt = DEFAULT_FORMATS.get(self.task_type, "")
        parts = [p for p in (role.strip(), self.prompt.strip(),
                             self.theory.strip(), fmt) if p]
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationConfig:
    """生成参数（与上游 Block 4 全局变量对应）。"""

    max_new_tokens: int = 50
    do_sample: bool = True
    temperature: float = 0.3
    top_p: float = 0.9
    seed: int | None = None
