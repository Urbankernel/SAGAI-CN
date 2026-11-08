"""支持 ``python -m sagai_cn`` 直接启动（等价于 ``sagai_cn`` 命令）。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
