# 纯 CPU 本地打分：GGUF + llama.cpp 环境搭建

本文档面向**没有 NVIDIA 显卡的电脑**，三步把本地打分跑起来：

1. 下载 `llama-server.exe`（llama.cpp 官方预编译二进制，免编译）
2. 下载 Qwen3-VL 的 GGUF 模型（语言权重 + mmproj 视觉投影，两个文件缺一不可）
3. 启动服务 → 验证单图能出分 → 跑 `sagai_cn infer --engine cpu`

> 为什么选这条路：GGUF 量化把模型压到几 GB，llama.cpp 对纯 CPU 做了深度优化，无GPU电脑也能在"一夜"内跑完一个片区的打分。相比云 API，它**数据不出本机、零计费**，代价是速度慢。

---

## 第 1 步：下载 llama-server

打开 [llama.cpp Releases](https://github.com/ggerganov/llama.cpp/releases)，
下载最新 release 中文件名含 **`win-cpu`** 的 zip（例如 `llama-bXXXX-bin-win-cpu-x64.zip`），解压后里面有 `llama-server.exe`。

> 要点：
> - 必须选 **cpu** 版（不是 cuda/cu 版，那些要显卡）。
> - 版本要够新：Qwen3-VL 架构需要较新的 llama.cpp 才支持。若你的模型跑不起来，
>   先升级 llama.cpp，或改用 `Qwen2.5-VL-7B-Instruct` 的 GGUF（更早被支持）。
> - 若 GitHub 下载慢，可用镜像站（hf-mirror.com）或国内代理。

## 第 2 步：下载 GGUF 模型（两个文件，必须配套）

以 **Qwen3-VL-8B-Instruct** 为例（官方仓库 `Qwen/Qwen3-VL-8B-Instruct-GGUF`）。
**推荐用工具自带命令下载**（默认走 ModelScope 魔搭，国内直连满速）：

```bash
sagai_cn download-models --gguf Qwen3-VL-8B-Instruct-GGUF
```

> ⚠️ 为什么不用 hf-mirror：Qwen 官方 GGUF 仓库在 HuggingFace 上是 **Xet 存储**，
> hf-mirror 会把请求重定向到海外 AWS CDN（`us.aws.cdn.hf.co`），国内下载会报
> `SSLError: UNEXPECTED_EOF_WHILE_READING`。ModelScope 有同名仓库、文件名一致，
> 走国内 `cdn-lfs-cn-1.modelscope.cn`，不跳海外。若确有需要，加
> `--gguf-source hf-mirror` 或 `--gguf-source hf` 可换源。

若手动下载，用 ModelScope 直链（文件名与命令一致）：

```
# 语言模型权重（Q4_K_M：体积小、速度快的推荐档）
https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/master/Qwen3VL-8B-Instruct-Q4_K_M.gguf

# 视觉投影 mmproj（必须与上面配套，建议 F16 不要过度量化）
https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/master/mmproj-Qwen3VL-8B-Instruct-F16.gguf
```

> 两个文件放到同一个文件夹（如 `models/gguf/`），**缺 mmproj 会直接报错**。
> 文件名以仓库实际文件列表为准，Q8_0 精度更高、体积更大（约 8.7 GB）。

可选模型档位（纯 CPU 经验值，8 核主流 CPU）：

| 档位 | GGUF 仓库 | 语言权重 | 内存占用 | 单图单任务速度 |
|---|---|---|---|---|
| 轻量 | Qwen/Qwen3-VL-2B-Instruct-GGUF | Q4_K_M | ~2–3 GB | 5–15 秒 |
| **推荐** | unsloth/Qwen3-VL-4B-Instruct-GGUF | Q4_K_M | ~4–5 GB | 15–40 秒 |
| 均衡 | Qwen/Qwen3-VL-8B-Instruct-GGUF | Q4_K_M | ~6–7 GB | 30–90 秒 |
| 高质量 | Qwen/Qwen3-VL-8B-Instruct-GGUF | Q8_0 | ~9 GB | 60–180 秒 |

## 第 3 步：启动 llama-server

在模型文件夹里打开终端（PowerShell / cmd 均可），运行：

```
llama-server.exe -m Qwen3VL-8B-Instruct-Q4_K_M.gguf ^
    --mmproj mmproj-Qwen3VL-8B-Instruct-F16.gguf ^
    --port 8080 --alias qwen3-vl --ctx-size 8192 ^
    -ngl 0 -t 8
```

参数说明：

| 参数 | 含义 |
|---|---|
| `-m` | 语言模型 GGUF 文件 |
| `--mmproj` | 视觉投影文件（多模态必需） |
| `--port 8080` | 服务端口，`--cpu-url` 默认对应它 |
| `--alias qwen3-vl` | 给模型起别名，`--cpu-model` 默认对应它 |
| `--ctx-size 8192` | 上下文长度；视觉 token 会占几千，要给足 |
| `-ngl 0` | 0 层放显卡 = **强制纯 CPU** |
| `-t 8` | 线程数，填你的 CPU 核数 |

看到 `server is listening on http://127.0.0.1:8080` 即启动成功。

## 第 4 步：验证单图能出分

浏览器打开 **http://localhost:8080**，上传一张街景 JPG，提问
"图中可见几棵树？只回答数字"，能返回数字即验证通过。

（可选）用命令行验证接口：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3-vl\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"描述这张图\"}]}],\"max_tokens\":100}"
```

## 第 5 步：跑打分

server 保持运行，另开一个终端回到项目根目录：

```bash
sagai_cn infer --engine cpu --tasks task/tasks.yaml --max-tokens 200
```

- 默认连 `http://127.0.0.1:8080/v1/chat/completions`、别名 `qwen3-vl`。
- 若端口/别名不同：加 `--cpu-url` / `--cpu-model` 覆盖。
- 结果仍是 `data/.../StreetViewBatchDownload/Score_Analysis_*.csv`，
  后续 `aggregate` / `map` 完全不变。

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 报"无法连接 llama-server" | server 没起 / 端口不对 | 回到第 3 步启动，检查端口 |
| 只下语言模型、没下 mmproj | 多模态需要双文件 | 补下 mmproj 再重启 |
| 模型加载报架构不支持 | llama.cpp 版本太旧 | 升级 llama.cpp，或换 Qwen2.5-VL-7B |
| 回答质量明显退化 | 缺 chat template / 图太大 | 确认用最新 llama-server；图片预缩到 1024–2048px |
| 内存不够（OOM） | 模型档位超出内存 | 换 4B 或 2B，见档位表 |
| 速度太慢 | 线程数没给足 / 模型太大 | `-t` 设核数；换 4B Q4_K_M |
