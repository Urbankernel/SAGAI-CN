"""Block 1 国产化：路网采样点生成。

三种路网来源（按优先级）：
  A. 本地矢量路网（shp/gpkg/geojson）—— 控规项目首选，数据最准
  B. 离线 .osm.pbf（pyrosm 解析）—— 稳定性最高，适合正式项目
  C. Overpass API（可配置国内可达镜像端点）—— 快速试验

采样算法与上游 SAGAI Block 1 一致：UTM 自动投影、highway 过滤、
端点偏移 + 等间距插值。
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

# 上游默认过滤集：去掉人行道/小径，避免与车行道重复采样
DEFAULT_HIGHWAY_TYPES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "unclassified", "service",
]


# ------------------------------------------------------------------
# 路网来源
# ------------------------------------------------------------------
def load_streets_from_vector(
    path: str | Path, polygon: Polygon | MultiPolygon | None = None,
) -> gpd.GeoDataFrame:
    """方案 A：读取本地矢量路网，映射到 OSM 字段契约。

    提供 polygon 时先按研究区粗过滤（剔除完全在区外的路段）；采样点级的
    精确裁剪（删除区外点）在 generate_points 中完成。polygon 为 None 时保留
    全部路网。highway/name 列缺失时自动补默认值。
    """
    streets = gpd.read_file(path)
    if streets.crs is None:
        raise ValueError(f"{path} 缺少坐标系定义，请先在外部指定 CRS")
    streets = streets.to_crs("EPSG:4326")
    if polygon is not None:
        streets = streets[streets.intersects(polygon)].copy()
    if "highway" not in streets.columns:
        streets["highway"] = "unclassified"  # 本地路网无等级字段时保底
    if "name" not in streets.columns:
        streets["name"] = ""
    streets["osmid"] = range(1, len(streets) + 1)
    return streets.reset_index(drop=True)


def load_streets_from_pbf(pbf_path: str | Path, polygon: Polygon) -> gpd.GeoDataFrame:
    """方案 B：pyrosm 解析离线 .osm.pbf（需 pip install pyrosm）。"""
    try:
        from pyrosm import OSM
    except ImportError as e:
        raise ImportError("离线 pbf 方案需要 pyrosm：pip install pyrosm") from e
    osm = OSM(str(pbf_path), bounding_box=polygon)
    _, edges = osm.get_network(network_type="all", nodes=True)
    edges = edges.to_crs("EPSG:4326")
    edges = edges[edges.intersects(polygon)].copy()
    if "highway" in edges.columns:
        edges["highway"] = edges["highway"].apply(
            lambda v: v[0] if isinstance(v, list) else v
        )
    return edges.reset_index(drop=True)


def load_streets_from_overpass(polygon: Polygon, overpass_url: str) -> gpd.GeoDataFrame:
    """方案 C：Overpass API（端点可换镜像，见 config.OVERPASS_URL）。"""
    import osmnx as ox

    ox.settings.timeout = 180
    ox.settings.overpass_rate_limit = True
    ox.settings.overpass_url = overpass_url
    graph = ox.graph_from_polygon(polygon, network_type="all")
    _, edges = ox.graph_to_gdfs(graph)
    edges = edges.reset_index()

    # 与上游一致：list 型属性转字符串，避免写 gpkg 报错
    def _join(v):
        return ", ".join(map(str, v)) if isinstance(v, list) else v

    for col in edges.columns:
        if any(isinstance(v, list) for v in edges[col]):
            edges[col] = edges[col].apply(_join)
    return edges.to_crs("EPSG:4326")


# ------------------------------------------------------------------
# 采样主流程（对应上游 run_block1）
# ------------------------------------------------------------------
def generate_points(
    polygon: Polygon | MultiPolygon | None,
    streets_gdf: gpd.GeoDataFrame,
    spacing_m: int = 40,
    offset_m: int = 15,
    highway_types: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """沿路网等间距生成采样点。返回 (points_gdf, streets_gdf)，均为 WGS84。

    polygon 提供时：① 用于确定 UTM 投影带；② 采样后按研究区裁剪——仅保留落在
    研究区内的采样点（删除区外点）。polygon 为 None 时改用路网 bbox 推断投影、
    不做裁剪。

    契约：streets 含 street_id/highway/name/length_m；
          points 含 point_id/street_id/Latitude/Longitude。
    """
    if polygon is not None and isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)

    streets = streets_gdf.copy()
    if highway_types:
        streets = streets[streets["highway"].isin(highway_types)].copy()
    if len(streets) == 0:
        raise ValueError("highway 过滤后路网为空，请放宽过滤条件或检查数据")

    # UTM 投影带：优先研究区，否则用路网 bbox 推断
    if polygon is not None:
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        projected_crs = polygon_gdf.estimate_utm_crs()
    else:
        projected_crs = streets.estimate_utm_crs()

    streets = streets.to_crs(projected_crs)
    streets["length_m"] = streets.geometry.length

    if "osmid" in streets.columns:
        streets["length_m"] = streets["length_m"].round(2)
        streets = streets.drop_duplicates(subset=["osmid", "length_m"], keep="first")

    keep = [c for c in ["geometry", "length_m", "osmid", "highway", "name"] if c in streets.columns]
    streets = streets[keep].reset_index(drop=True)
    streets["street_id"] = [f"street_{i + 1}" for i in range(len(streets))]

    # 与上游一致的插值逻辑：两端各留 offset，中间按 spacing 等距
    all_points: list[dict] = []
    counter = 1
    for _, row in streets.iterrows():
        line = row.geometry
        length = line.length
        sid = row["street_id"]
        if length < offset_m:
            continue
        elif length < offset_m + spacing_m:
            positions = [length / 2]
        else:
            positions = []
            current = float(offset_m)
            while current <= length - offset_m:
                positions.append(current)
                current += spacing_m
        for dist in positions:
            all_points.append(
                {"geometry": line.interpolate(dist), "point_id": f"point_{counter}", "street_id": sid}
            )
            counter += 1

    if not all_points:
        raise ValueError("未生成任何采样点：路网过短或 offset/spacing 过大")

    points = gpd.GeoDataFrame(all_points, crs=projected_crs)
    points_wgs = points.to_crs("EPSG:4326")
    points_wgs["Latitude"] = points_wgs.geometry.y
    points_wgs["Longitude"] = points_wgs.geometry.x

    # 点级裁剪：仅保留研究区内的采样点（含边界），删除区外点
    if polygon is not None:
        n_before = len(points_wgs)
        points_wgs = points_wgs[points_wgs.geometry.intersects(polygon)].copy()
        n_clipped = n_before - len(points_wgs)
        if n_clipped:
            print(f"  按研究区裁剪：删除研究区外采样点 {n_clipped} 个，"
                  f"保留 {len(points_wgs)} 个")

    return points_wgs, streets.to_crs("EPSG:4326")


def save_case_gpkg(points: gpd.GeoDataFrame, streets: gpd.GeoDataFrame, gpkg_path: Path) -> None:
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    streets.to_file(gpkg_path, layer="streets", driver="GPKG")
    points.to_file(gpkg_path, layer="points", driver="GPKG")
