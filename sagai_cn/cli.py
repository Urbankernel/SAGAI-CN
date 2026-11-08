"""SAGAI-CN 命令行入口。

子命令对应原 Colab 各 Block：
  sample      Block 1  路网采样点生成
  download    Block 2  百度全景批量下载（默认逆向接口，无需 AK；可切官方 API）
  ingest      Block 2  替代：导入用户自采照片（EXIF 定位，推荐）
  infer       Block 3-5  模型打分（local=UVLM 本地 / dashscope=云 API）
  aggregate   Block 6  空间聚合 + joined gpkg
  map         Block 6  天地图底图交互 HTML 地图
  download-models      预下载 ModelScope 权重

示例（最小全流程）：
  # 数据默认落在 --area 或 --roads 所在文件夹（如 data/SH/）；也可用 --base 或 SAGAI_BASE 覆盖
  # --area 可选：提供时采样后仅保留研究区内点位（删除区外点），省略则沿整条路网布点
  sagai_cn sample --area data/SH/study_area.gpkg --roads data/SH/roads.gpkg
  sagai_cn download
  sagai_cn ingest --photos ./photos
  sagai_cn infer --engine dashscope --tasks task/tasks.yaml
  sagai_cn aggregate --score-col green_view
  sagai_cn map --score-col green_view
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config


def _require_file(path: Path, what: str) -> Path:
    """输入文件前置检查：找不到时给出人话提示和候选文件，而不是堆栈报错。"""
    if path.exists():
        return path
    cwd = Path.cwd()
    # 只在当前文件夹及其下一层找候选，避免扫描 models/ 等大目录
    candidates = []
    for folder in (cwd, *[d for d in cwd.iterdir() if d.is_dir()
                           and d.name not in ("models", "vendor", ".git", "__pycache__")]):
        exts = [path.suffix] if path.suffix else []
        candidates += [f for ext in exts for f in sorted(folder.glob(f"*{ext}"))]
        candidates += sorted(folder.glob(path.name))
    if not candidates:  # 后缀都打错了（如把 gpkg 写成 geojson）时，放宽到常见矢量格式再找一遍
        for folder in (cwd, *[d for d in cwd.iterdir() if d.is_dir()
                               and d.name not in ("models", "vendor", ".git", "__pycache__")]):
            candidates += [f for ext in (".gpkg", ".geojson", ".shp", ".json")
                           for f in sorted(folder.glob(f"*{ext}"))]
    seen, uniq = set(), []
    for c in candidates:
        if c not in seen and c.is_file():
            seen.add(c)
            uniq.append(c)
    lines = [
        f"找不到{what}文件：{path}",
        f"当前所在文件夹：{cwd}",
        "排查提示：",
        "  1. 指令里的文件名必须和实际文件一字不差（注意 .geojson 和 .gpkg 后缀不同）；",
        "  2. 相对路径以『当前所在文件夹』为准，文件在子文件夹时要带上，如 data\\xxx.gpkg；",
        "  3. 也可以用完整路径，如 F:\\data\\xxx.gpkg。",
    ]
    if uniq:
        lines.append("在附近找到了这些可能是你要用的文件：")
        lines += [f"  - {c.relative_to(cwd)}" for c in uniq[:10]]
    raise SystemExit("\n".join(lines))


# map 命令自动发现路网/研究范围时的候选文件名（按优先级）
_ROADS_NAMES = ["roads.gpkg", "roads.geojson", "roads.shp"]
_AREA_NAMES = ["study_area.gpkg", "study_area.geojson", "study_area.shp"]


def _find_aux_file(base_dir: Path, candidates: list[str]) -> Path | None:
    """在数据目录下按候选文件名找第一个存在的文件。找不到返回 None。"""
    for name in candidates:
        p = base_dir / name
        if p.is_file():
            return p
    return None


def _discover_base_dir(default: Path) -> tuple[Path | None, list[Path]]:
    """在默认数据根目录的直接子目录里，找含 street_samples.gpkg 的目录。

    返回 (唯一命中目录 | None, 全部命中目录)。用于后续命令自动定位上次
    sample 落地的数据目录，实现"一个数据根目录 = 一个项目"的免参数体验。
    """
    hits: list[Path] = []
    if default.is_dir():
        hits = [d for d in sorted(default.iterdir())
                if d.is_dir() and (d / "StreetSamples" / "street_samples.gpkg").exists()]
    return (hits[0] if len(hits) == 1 else None), hits


def _resolve_base_dir(args) -> Path:
    """确定数据根目录 base_dir。

    优先级：
      1. ``--base`` 参数（所有子命令，显式覆盖）
      2. ``sample`` 命令：``--area`` 所在文件夹，无 ``--area`` 时退 ``--roads``
         所在文件夹（路网/研究区放哪，产物就跟到哪）
      3. 其余命令：自动发现 ``./data`` 下唯一的 case 子目录
      4. ``SAGAI_BASE`` 环境变量（.env 默认 ``./data``，作为兜底）
    """
    if getattr(args, "base", None):
        return Path(args.base)

    if getattr(args, "cmd", None) == "sample":
        anchor = getattr(args, "area", None) or getattr(args, "roads", None)
        if anchor:
            parent = Path(anchor).parent
            if str(parent) not in ("", "."):
                return parent
        # 无 area/roads（Overpass 源）或锚点就在当前目录：退回 SAGAI_BASE 或 ./data
        env = os.environ.get("SAGAI_BASE")
        return Path(env) if env else Path("./data")

    default = Path("./data")
    discovered, hits = _discover_base_dir(default)
    if discovered:
        return discovered
    if len(hits) > 1:
        print("检测到多个数据目录，请用 --base 指定其一：")
        for h in hits:
            print(f"  - {h}")

    env = os.environ.get("SAGAI_BASE")
    return Path(env) if env else default


def _load_task_defs(path: Path) -> list[dict]:
    """读取任务定义文件，按后缀自动识别 JSON / YAML。

    YAML 支持 ``#`` 注释（用于给键值对写解释文字）；JSON 无注释但体积更小。
    兼容顶部 ``tasks: [...]`` 包裹结构（旧模板库格式）。
    """
    text = _require_file(path, "任务定义").read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise SystemExit(
                "读取 YAML 任务定义需要 PyYAML，请先安装：pip install PyYAML") from e
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "tasks" in data:
            data = data["tasks"]
        return data or []
    return json.loads(text)


def _read_polygon(path: Path):
    import geopandas as gpd
    from shapely.ops import unary_union

    gdf = gpd.read_file(path)
    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError(f"{path} 中无有效几何")
    geom = unary_union(gdf.geometry)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def cmd_sample(cfg: Config, args) -> None:
    from . import sampling

    polygon = None
    if args.area:
        polygon = _read_polygon(_require_file(Path(args.area), "研究区"))
        print(f"研究区：{args.area}（采样后仅保留研究区内点位）")

    if args.roads:
        print(f"路网来源：本地矢量 {args.roads}")
        streets = sampling.load_streets_from_vector(
            _require_file(Path(args.roads), "路网"), polygon)
    elif args.pbf:
        if polygon is None:
            raise SystemExit("使用离线 pbf 路网时必须提供 --area（用于界定研究范围）")
        print(f"路网来源：离线 pbf {args.pbf}")
        streets = sampling.load_streets_from_pbf(
            _require_file(Path(args.pbf), "离线路网"), polygon)
    else:
        if polygon is None:
            raise SystemExit("使用 Overpass 路网时必须提供 --area（用于界定研究范围）")
        print(f"路网来源：Overpass {cfg.overpass_url}")
        streets = sampling.load_streets_from_overpass(polygon, cfg.overpass_url)

    points, streets = sampling.generate_points(
        polygon=polygon, streets_gdf=streets,
        spacing_m=args.spacing, offset_m=args.offset,
        highway_types=args.highway or None,
    )
    cfg.ensure_dirs()
    gpkg = cfg.gpkg_path()
    sampling.save_case_gpkg(points, streets, gpkg)
    # 导出采样点坐标 CSV（name,lng,lat，WGS84），供逆向脚本/人工核点使用
    coords_csv = cfg.samples_dir / "street_samples_points.csv"
    points[["point_id", "Longitude", "Latitude"]].rename(
        columns={"point_id": "name", "Longitude": "lng", "Latitude": "lat"}
    ).to_csv(coords_csv, index=False, encoding="utf-8-sig")
    print(f"采样完成：{len(points)} 点 / {len(streets)} 街段 -> {gpkg}")
    print(f"坐标清单：{coords_csv}（name,lng,lat）")


def cmd_download(cfg: Config, args) -> None:
    if args.source == "reverse":
        from .pano_reverse import download_streetview_reverse

        download_streetview_reverse(
            gpkg_path=cfg.gpkg_path(),
            download_dir=cfg.images_dir(),
            level=args.pano_level,
            sleep_seconds=args.interval,
            max_sub_m=args.max_sub,
            crop_bottom=args.crop_bottom,
            crop_top=args.crop_top,
            search_radius=args.search_radius,
        )
        return

    from .streetview import BaiduPanoClient, download_streetview

    client = BaiduPanoClient(
        aks=cfg.baidu_aks, base_url=cfg.baidu_pano_url,
        width=args.width, height=args.height,
        request_interval=args.interval,
    )
    download_streetview(
        gpkg_path=cfg.gpkg_path(),
        download_dir=cfg.images_dir(),
        client=client, mode=args.mode, pitch=args.pitch, fov=args.fov,
        max_sub_m=args.max_sub, dedup_by_hash=not args.no_dedup,
    )


def cmd_ingest(cfg: Config, args) -> None:
    from .ingest import ingest_photos

    ingest_photos(
        gpkg_path=cfg.gpkg_path(),
        photos_dir=_require_file(Path(args.photos), "照片文件夹"),
        download_dir=cfg.images_dir(),
        snap_max_m=args.snap_max,
        manual_csv=Path(args.manual) if args.manual else None,
        tianditu_tk=cfg.tianditu_tk,
    )


def cmd_infer(cfg: Config, args) -> None:
    from .inference import CpuEngine, DashScopeEngine, LocalEngine
    from .tasks import GenerationConfig, TaskSpec

    task_defs = _load_task_defs(Path(args.tasks))
    tasks = [TaskSpec(**t) for t in task_defs]
    gen = GenerationConfig(max_new_tokens=args.max_tokens,
                           temperature=args.temperature, top_p=args.top_p,
                           seed=args.seed)
    image_dir = cfg.images_dir()
    out_csv = image_dir / f"Score_Analysis_{args.model_tag}.csv"

    if args.engine == "dashscope":
        engine = DashScopeEngine(api_key=cfg.dashscope_api_key,
                                 model=args.api_model,
                                 base_url=cfg.dashscope_base_url, rpm=args.rpm)
        engine.run_batch(tasks, image_dir, out_csv, gen, role=args.role)
    elif args.engine == "cpu":
        engine = CpuEngine(base_url=args.cpu_url, model=args.cpu_model)
        engine.run_batch(tasks, image_dir, out_csv, gen, role=args.role)
    else:
        from .models import download_model, load_model_uvlm

        model_path = download_model(args.model, cfg.model_dir)
        ctx = load_model_uvlm(model_path, precision=args.precision)
        LocalEngine(ctx).run_batch(tasks, image_dir, out_csv, gen)


def cmd_aggregate(cfg: Config, args) -> None:
    from .aggregate import aggregate

    image_dir = cfg.images_dir()
    csvs = sorted(image_dir.glob("Score_Analysis_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"{image_dir} 下无 Score_Analysis_*.csv，请先 infer")
    if len(csvs) > 1:
        print(f"发现 {len(csvs)} 个评分 CSV，使用最新：{csvs[0].name}")
    # 结果文件带 score_col 后缀，避免多个 task 的结果互相覆盖
    out = cfg.results_dir / f"joined_{args.score_col}.gpkg"
    aggregate(cfg.gpkg_path(), csvs[0], out,
              score_column=args.score_col, view_filter=args.view_filter)


def cmd_map(cfg: Config, args) -> None:
    from .mapping import build_points_map, build_streets_map
    import geopandas as gpd

    joined = cfg.results_dir / f"joined_{args.score_col}.gpkg"
    points = gpd.read_file(joined, layer="points")
    streets = gpd.read_file(joined, layer="streets")
    pt_col = {"average": "mean", "sum": "sum", "median": "median"}[args.agg]
    st_col = {"average": "avg_score", "sum": "total_score",
              "median": "median_score"}[args.agg]

    image_dir = cfg.images_dir()
    csvs = sorted(image_dir.glob("Score_Analysis_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)

    # 路网 / 研究范围：显式 --roads/--area 优先，否则在数据目录下自动发现同名文件
    roads_src = (Path(args.roads) if args.roads
                 else _find_aux_file(cfg.base_dir, _ROADS_NAMES))
    area_src = (Path(args.area) if args.area
                else _find_aux_file(cfg.base_dir, _AREA_NAMES))
    roads = area = None
    if roads_src:
        roads = gpd.read_file(_require_file(roads_src, "路网"))
        print(f"路网图层：{roads_src.name}")
    if area_src:
        area = gpd.read_file(_require_file(area_src, "研究范围"))
        print(f"研究范围图层：{area_src.name}")

    pmap = build_points_map(points, csvs[0] if csvs else None,
                            args.score_col, pt_col, image_dir,
                            cfg.tianditu_tk, args.cmap,
                            roads=roads, area=area)
    p1 = cfg.results_dir / f"points_map_{args.score_col}.html"
    pmap.save(p1)
    smap = build_streets_map(streets, st_col, cfg.tianditu_tk, args.cmap)
    p2 = cfg.results_dir / f"streets_map_{args.score_col}.html"
    smap.save(p2)
    print(f"地图已生成：\n  {p1}\n  {p2}")


def cmd_download_models(cfg: Config, args) -> None:
    from .models import MODEL_REGISTRY, download_gguf, download_model

    if args.gguf:
        for name in args.gguf:
            download_gguf(name, cfg.model_dir, quant=args.gguf_quant,
                          source=args.gguf_source)
    elif args.models:
        for name in args.models:
            download_model(name, cfg.model_dir)
    else:
        for name in MODEL_REGISTRY:
            download_model(name, cfg.model_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sagai_cn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--base",
        help="数据根目录（默认：sample 取 --area 或 --roads 所在文件夹；其余命令"
             "自动发现 data 下唯一 case 目录，或读 SAGAI_BASE）")

    sp = sub.add_parser("sample", help="Block 1 路网采样", parents=[common])
    sp.add_argument("--area", help="研究区矢量文件 gpkg/geojson/shp（可选；提供时采样后"
                                   "仅保留研究区内点位。用 --roads 本地路网时可省略；"
                                   "--pbf / Overpass 源时必需）")
    src = sp.add_mutually_exclusive_group()
    src.add_argument("--roads", help="本地路网矢量文件（推荐）")
    src.add_argument("--pbf", help="离线 .osm.pbf 文件")
    sp.add_argument("--spacing", type=int, default=40)
    sp.add_argument("--offset", type=int, default=15)
    sp.add_argument("--highway", nargs="*", default=None,
                    help="highway 类型过滤，默认全部保留")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("download", help="Block 2 百度全景下载", parents=[common])
    sp.add_argument("--source", choices=["reverse", "api"], default="reverse",
                    help="reverse=逆向接口（无需 AK，推荐，高清全景切四视角）；"
                         "api=官方全景静态图 API（需 BAIDU_AK）")
    sp.add_argument("--pano-level", type=int, choices=[1, 2, 3, 4], default=3,
                    help="reverse 源全景清晰度 1~4（默认 3=4096×2048；"
                         "4=8192×4096，请求量/存储/风控风险 4 倍，VLM 会预缩放故无精度收益）")
    sp.add_argument("--mode", choices=["4_views", "left_right", "front_back"],
                    default="4_views", help="仅 api 源生效")
    sp.add_argument("--pitch", type=int, default=-20, help="仅 api 源生效")
    sp.add_argument("--fov", type=int, default=90, help="仅 api 源生效")
    sp.add_argument("--max-sub", type=float, default=50.0)
    sp.add_argument("--width", type=int, default=640, help="仅 api 源生效")
    sp.add_argument("--height", type=int, default=640, help="仅 api 源生效")
    sp.add_argument("--interval", type=float, default=0.2,
                    help="请求间隔秒（reverse 为瓦片间隔，api 为请求限速）")
    sp.add_argument("--crop-bottom", type=float, default=0.4,
                    help="reverse 源：裁掉全景底部比例（默认 0.4 = 去掉下方 40%% 的"
                         "采集车顶/车头）；设 0 关闭")
    sp.add_argument("--crop-top", type=float, default=0.0,
                    help="reverse 源：裁掉全景顶部比例（默认 0 = 不裁天空）；"
                         "与 --crop-bottom 之和必须小于 1")
    sp.add_argument("--search-radius", type=float, default=5.0,
                    help="reverse 源：坐标本身无街景时的就近搜索半径（米，默认 5）；"
                         "设 0 关闭就近搜索")
    sp.add_argument("--no-dedup", action="store_true",
                    help="仅 api 源生效：关闭内容哈希去重")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("ingest", help="Block 2 替代：导入用户自采照片（EXIF 定位）",
                        parents=[common])
    sp.add_argument("--photos", required=True, help="照片文件夹路径")
    sp.add_argument("--snap-max", type=float, default=50.0,
                    help="照片归属采样点的最大距离（米），超过判为不匹配")
    sp.add_argument("--manual", help="人工补录表 manual_coords.csv（无 GPS 照片用）")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("infer", help="Block 3-5 VLM 批量打分", parents=[common])
    sp.add_argument("--tasks", required=True, help="任务定义文件（JSON 或 YAML，TaskSpec 列表）")
    sp.add_argument("--engine", choices=["local", "dashscope", "cpu"], default="local")
    sp.add_argument("--model", default="Qwen3-VL-8B-Instruct",
                    help="local 引擎：MODEL_REGISTRY 中的模型名")
    sp.add_argument("--api-model", default="qwen3-vl-plus",
                    help="dashscope 引擎：API 模型名")
    sp.add_argument("--cpu-url", default="http://127.0.0.1:8080/v1/chat/completions",
                    help="cpu 引擎：llama-server 的 OpenAI 兼容端点")
    sp.add_argument("--cpu-model", default="qwen3-vl",
                    help="cpu 引擎：llama-server 启动时的 --alias 别名")
    sp.add_argument("--model-tag", default=None, help="输出 CSV 后缀，默认按引擎/模型生成")
    sp.add_argument("--precision", choices=["4bit", "8bit", "fp16"], default="4bit")
    sp.add_argument("--max-tokens", type=int, default=50)
    sp.add_argument("--temperature", type=float, default=0.3)
    sp.add_argument("--top-p", type=float, default=0.9)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--role", default="", help="所有任务共享的角色指令")
    sp.add_argument("--rpm", type=int, default=60, help="dashscope 限流（请求/分钟）")
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser("aggregate", help="Block 6 空间聚合", parents=[common])
    sp.add_argument("--score-col", required=True, help="CSV 中的任务列名，如 green_view")
    sp.add_argument("--view-filter", default="all",
                    choices=["all", "left", "right", "front", "back",
                             "left_right", "front_back"])
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("map", help="Block 6 交互地图", parents=[common])
    sp.add_argument("--score-col", required=True)
    sp.add_argument("--agg", choices=["average", "sum", "median"], default="average")
    sp.add_argument("--cmap", default="viridis")
    sp.add_argument("--roads", help="路网矢量文件；不传则自动发现数据目录下的 "
                                   "roads.gpkg / roads.geojson / roads.shp")
    sp.add_argument("--area", help="研究范围矢量文件；不传则自动发现数据目录下的 "
                                  "study_area.gpkg / study_area.geojson / study_area.shp")
    sp.set_defaults(func=cmd_map)

    sp = sub.add_parser("download-models", help="预下载 ModelScope 权重")
    sp.add_argument("--models", nargs="*", default=None, help="默认下载全部内置模型")
    sp.add_argument("--gguf", nargs="*", default=None,
                    help="GGUF 模型名（纯 CPU 打分，见 GGUF_REGISTRY）")
    sp.add_argument("--gguf-quant", default="q4_k_m",
                    help="GGUF 量化档：q4_k_m / q5_k_m / q8_0 / f16")
    sp.add_argument("--gguf-source", default="modelscope",
                    choices=["modelscope", "hf-mirror", "hf"],
                    help="GGUF 下载源（默认 modelscope，国内直连；hf-mirror 对 "
                         "Xet 仓库会重定向海外 CDN，可能 SSL 失败）")
    sp.set_defaults(func=cmd_download_models)

    args = p.parse_args(argv)
    # model-tag 默认规则：与上游 per-model CSV 约定一致
    if getattr(args, "cmd", None) == "infer" and not args.model_tag:
        tag = (args.api_model if args.engine == "dashscope"
               else args.cpu_model if args.engine == "cpu" else args.model)
        args.model_tag = tag.replace("/", "_").replace(".", "_")
    cfg = Config(base_dir=_resolve_base_dir(args))
    cfg.ensure_dirs()
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
