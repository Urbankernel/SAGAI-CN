"""Block 6 国产化：分数空间聚合。

逻辑与上游 Block 6 一致：从文件名解析 point_id/视角 -> 按视角过滤 ->
聚合到点级（mean/sum/median/var + 分视角均值）-> 再聚合到街道级 ->
输出 joined GeoPackage（EPSG:4326）。
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

VIEW_SUFFIXES = ("left", "right", "front", "back")

# 采样点 point_id 的格式：point_ 前缀 + 数字（如 point_12）
_POINT_ID_RE = re.compile(r"^point_\d+")


def extract_base_id(image_name: str) -> str:
    """从文件名提取 point_id。

    point_12_left.jpg -> point_12
    point_12_left_NA.jpg -> point_12
    point_1_归昌路_20190519_left.jpg -> point_1（逆向下载带路名/日期的命名）
    """
    base = Path(image_name).stem.replace("_NA", "")
    m = _POINT_ID_RE.match(base)
    if m:
        return m.group(0)
    # 非 point_ 前缀的兜底：去掉最后一段（视角后缀）
    parts = base.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 3 else base


def extract_view_suffix(image_name: str) -> str:
    base = Path(image_name).stem.replace("_NA", "")
    parts = base.split("_")
    return parts[-1] if len(parts) >= 3 else ""


def detect_image_mode(csv_df: pd.DataFrame) -> tuple[str, int]:
    suffixes = {extract_view_suffix(n) for n in csv_df["image_name"]}
    suffixes.discard("")
    if set(VIEW_SUFFIXES).issubset(suffixes):
        return "4_views", 4
    if {"left", "right"}.issubset(suffixes):
        return "left_right", 2
    if {"front", "back"}.issubset(suffixes):
        return "front_back", 2
    return "unknown", max(len(suffixes), 1)


def filter_views(df: pd.DataFrame, view_filter: str) -> pd.DataFrame:
    if view_filter == "all":
        return df
    if view_filter in VIEW_SUFFIXES:
        return df[df["view"] == view_filter]
    if view_filter == "left_right":
        return df[df["view"].isin(["left", "right"])]
    if view_filter == "front_back":
        return df[df["view"].isin(["front", "back"])]
    raise ValueError(f"未知 view_filter: {view_filter}")


def aggregate(
    gpkg_path: Path,
    csv_path: Path,
    output_gpkg: Path,
    score_column: str,
    view_filter: str = "all",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """聚合主流程。返回 (points_joined, streets_joined)，并写 output_gpkg。"""
    points = gpd.read_file(gpkg_path, layer="points")
    streets = gpd.read_file(gpkg_path, layer="streets")
    scores = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

    if score_column not in scores.columns:
        avail = [c for c in scores.columns
                 if c != "image_name" and not c.endswith(("_raw", "_reasoning", "_truncated"))]
        raise KeyError(f"列 {score_column} 不存在，可选：{avail}")

    scores["base_id"] = scores["image_name"].apply(extract_base_id)
    scores["view"] = scores["image_name"].apply(extract_view_suffix)
    scores["score"] = pd.to_numeric(
        scores[score_column].replace("NA", pd.NA), errors="coerce")

    # 非数值列（category/boolean 任务）无法聚合——转数值后全空时明确报错，
    # 而不是静默产出全空评分列，让用户误以为数据丢失。
    if scores["score"].notna().sum() == 0:
        numeric_cols = []
        for c in scores.columns:
            if c == "image_name" or c.endswith(("_raw", "_reasoning", "_truncated")):
                continue
            try:
                if pd.to_numeric(scores[c].replace("NA", pd.NA),
                                 errors="coerce").notna().any():
                    numeric_cols.append(c)
            except Exception:
                pass
        sample = scores[score_column].dropna().unique()[:5].tolist()
        raise ValueError(
            f"评分列 {score_column} 不是数值列（样例取值 {sample}），"
            f"无法聚合。请改用数值型任务列（如 {numeric_cols or '无'}），"
            f"再重跑 aggregate 和 map。")

    filtered = filter_views(scores, view_filter)
    mode, n_img = detect_image_mode(scores)
    print(f"影像模式 {mode}（{n_img} 视角/点），视角过滤 {view_filter}："
          f"{len(filtered)}/{len(scores)} 张参与聚合")

    # ---- 点级聚合：mean/sum/median/var + 分视角均值（与上游一致，全量统计）----
    all_ids = points[["point_id"]].rename(columns={"point_id": "base_id"})
    merged = all_ids.merge(filtered[["base_id", "score"]], on="base_id", how="left")
    stats = merged.groupby("base_id", dropna=False)["score"].agg(
        mean="mean", sum="sum", median="median", var="var").reset_index()
    points = points.merge(stats, left_on="point_id", right_on="base_id",
                          how="left").drop(columns="base_id")

    # 分视角均值（基于未过滤数据，保持上游 v2.1 行为）
    for view in VIEW_SUFFIXES:
        vdf = scores[scores["view"] == view]
        vmean = vdf.groupby("base_id")["score"].mean().rename(f"{score_column}_{view}")
        points = points.merge(vmean, left_on="point_id", right_index=True, how="left")

    # ---- 街道级二次聚合 ----
    valid = points[points["mean"].notna()]
    street_stats = valid.groupby("street_id").agg(
        avg_score=("mean", "mean"), total_score=("sum", "sum"),
        median_score=("median", "mean"), var_score=("var", "mean"),
    ).reset_index()
    streets = streets.merge(street_stats, on="street_id", how="left")

    points = points.to_crs("EPSG:4326")
    streets = streets.to_crs("EPSG:4326")
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    points.to_file(output_gpkg, layer="points", driver="GPKG")
    streets.to_file(output_gpkg, layer="streets", driver="GPKG")
    print(f"已写出：{output_gpkg}（points {len(points)} / streets {len(streets)}）")
    return points, streets
