"""Block 2 替代数据源：用户自采照片导入（替代百度全景 API 下载）。

背景：百度全景静态图 API 已转为高级功能，个人开发者无法调用。
本模块把"街景获取"改为用户自行上传照片：

  1. 批量读取照片 EXIF，提取 GPS 经纬度（手机/相机拍摄自带）；
  2. 将照片吸附到最近的采样点（沿用 Block 1 生成的 gpkg），
     按 {point_id}_{view}.jpg 契约重命名存入影像目录——
     下游 infer/aggregate/map 对照片来源无感，零改动；
  3. 视角判定：EXIF 有拍摄方向(GPSImgDirection)时，结合点位街道走向
     归为 left/right/front/back 中最近的一个；无方向信息时按空位分配；
  4. 缺 GPS / 超距 / 无法解析的照片逐张反馈，支持两种补录方式：
     - 手动填写坐标（manual_coords.csv 填 lat/lon）
     - 地图选点（打开生成的选点地图查 point_id，填进同一 CSV）

坐标纪律：EXIF 坐标视为 WGS84（手机相机标准）；吸附只用于归属判定，
成果点位的几何不变（与百度方案"不做吸附回写"一致）。
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PIL import Image as PILImage

from .coords import haversine_m
from .streetview import (compute_bearing_for_point, get_headings_for_mode,
                         VIEW_SUFFIXES)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
HEIC_EXTS = {".heic", ".heif"}  # 需 pillow-heif，未装时给明确提示
# 无 EXIF 方向时，视角按此顺序填空位（立面分析优先左右）
VIEW_FILL_ORDER = ("left", "right", "front", "back")

GPS_IFD = 0x8825
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4
_GPS_IMG_DIRECTION = 17  # EXIF 拍摄方向（16 是 DirectionRef，勿混）


# ------------------------------------------------------------------
# EXIF 解析
# ------------------------------------------------------------------
def _dms_to_deg(dms, ref: str) -> float | None:
    """EXIF 度分秒 -> 十进制度。ref 为 N/S/E/W。"""
    try:
        d, m, s = (float(v) for v in dms)
    except Exception:
        return None
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def read_exif_gps(path: Path) -> dict | None:
    """读取照片 EXIF 的 GPS 信息。

    返回 {"lat":…, "lon":…, "direction":…|None}；无 GPS 返回 None；
    文件无法解析时抛异常（由调用方归类为 bad_file）。
    """
    with PILImage.open(path) as img:
        exif = img.getexif()
        if not exif:
            return None
        gps = exif.get_ifd(GPS_IFD)
        if not gps:
            return None
        lat = _dms_to_deg(gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF, "N")) \
            if gps.get(_GPS_LAT) else None
        lon = _dms_to_deg(gps.get(_GPS_LON), gps.get(_GPS_LON_REF, "E")) \
            if gps.get(_GPS_LON) else None
        if lat is None or lon is None:
            return None
        direction = gps.get(_GPS_IMG_DIRECTION)
        return {"lat": lat, "lon": lon,
                "direction": float(direction) if direction is not None else None}


# ------------------------------------------------------------------
# 视角与吸附
# ------------------------------------------------------------------
def _angular_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def classify_view(direction: float | None, bearing: float,
                  taken: set[str]) -> str | None:
    """确定照片视角后缀。

    有拍摄方向：归为 left/right/front/back 中角度最近且未被占用的一个；
    无方向：按 VIEW_FILL_ORDER 取第一个空位；全满返回 None（判为重复）。
    """
    candidates = get_headings_for_mode("4_views", bearing)
    if direction is not None:
        for heading, suffix in sorted(candidates,
                                      key=lambda hs: _angular_diff(direction, hs[0])):
            if suffix not in taken:
                return suffix
        return None
    for suffix in VIEW_FILL_ORDER:
        if suffix not in taken:
            return suffix
    return None


@dataclass
class _PointIndex:
    """采样点索引：经纬度 + 街道走向（用于吸附与视角判定）。"""

    gdf: gpd.GeoDataFrame
    bearings: dict[str, float]

    @classmethod
    def from_gpkg(cls, gpkg_path: Path) -> "_PointIndex":
        points = gpd.read_file(gpkg_path, layer="points")
        streets = gpd.read_file(gpkg_path, layer="streets")
        proj = points.estimate_utm_crs()
        pts_p = points.to_crs(proj)
        sts_p = streets.to_crs(proj)
        lookup = dict(zip(sts_p["street_id"], sts_p.geometry))
        bearings = {}
        for _, row in pts_p.iterrows():
            geom = lookup.get(row.get("street_id"))
            bearings[row["point_id"]] = (
                compute_bearing_for_point(row.geometry, geom) if geom is not None else 0.0)
        return cls(gdf=points, bearings=bearings)

    def nearest(self, lat: float, lon: float) -> tuple[str, float] | None:
        """最近采样点及球面距离（米）。无点返回 None。"""
        best_pid, best_d = None, math.inf
        for _, row in self.gdf.iterrows():
            d = haversine_m(lat, lon, row["Latitude"], row["Longitude"])
            if d < best_d:
                best_pid, best_d = row["point_id"], d
        return (best_pid, best_d) if best_pid is not None else None


# ------------------------------------------------------------------
# 人工补录（手动坐标 / 地图选点）
# ------------------------------------------------------------------
def _load_manual(manual_csv: Path) -> dict[str, dict]:
    """读取人工补录表。列：filename, [lat], [lon], [point_id]。

    lat/lon 与 point_id 至少给一样；point_id 优先（地图选点方式）。
    """
    df = pd.read_csv(manual_csv, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if "filename" not in df.columns:
        raise ValueError("补录表必须包含 filename 列（照片文件名）")
    table = {}
    for _, row in df.iterrows():
        fn = str(row["filename"]).strip()
        if not fn or fn == "nan":
            continue
        entry = {}
        if "point_id" in df.columns and pd.notna(row.get("point_id")):
            pid = str(row["point_id"]).strip()
            if pid:
                entry["point_id"] = pid
        try:
            if "lat" in df.columns and "lon" in df.columns \
                    and pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                entry["lat"], entry["lon"] = float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError):
            pass
        if entry:
            table[fn] = entry
    return table


def _write_manual_template(path: Path, filenames: list[str]) -> None:
    """生成待补录 CSV 模板，用户填完后以 --manual 重新导入。"""
    pd.DataFrame({"filename": filenames, "lat": "", "lon": "",
                  "point_id": ""}).to_csv(path, index=False, encoding="utf-8-sig")


def _build_pick_map(index: _PointIndex, out_html: Path, tk: str) -> None:
    """生成选点地图：展示全部采样点及 point_id，供无 GPS 照片人工对照。"""
    import folium
    from .mapping import _base_map

    gdf = index.gdf
    m = _base_map(gdf["Latitude"].mean(), gdf["Longitude"].mean(), tk, zoom=15)
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]], radius=6,
            color="#e74c3c", fill=True, fill_opacity=0.8,
            popup=f"<b>{row['point_id']}</b><br>{row.get('street_id', '')}",
            tooltip=row["point_id"],
        ).add_to(m)
    m.save(out_html)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def ingest_photos(
    gpkg_path: Path,
    photos_dir: Path,
    download_dir: Path,
    snap_max_m: float = 50.0,
    manual_csv: Path | None = None,
    tianditu_tk: str = "",
) -> pd.DataFrame:
    """批量导入用户照片。返回逐张处理报告（并写 ingest_report.csv）。

    状态机：ok / ok_manual / no_gps / unmatched / duplicate /
            bad_file / unsupported / skipped_manual_missing
    """
    if not photos_dir.is_dir():
        raise SystemExit(f"照片文件夹不存在：{photos_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)
    index = _PointIndex.from_gpkg(gpkg_path)
    if len(index.gdf) == 0:
        raise SystemExit("采样点为空：请先运行 sample 生成点位")

    manual = _load_manual(manual_csv) if manual_csv else {}
    taken: dict[str, set[str]] = {}  # point_id -> 已占用视角
    # 已存在的影像也算占用（断点续导）
    for f in download_dir.glob("point_*_*.jpg"):
        stem = f.stem.replace("_NA", "")
        parts = stem.split("_")
        if len(parts) >= 3:
            taken.setdefault("_".join(parts[:-1]), set()).add(parts[-1])

    # 幂等：上一轮已成功导入的源文件直接跳过（不重新占视角、不覆盖）
    already: dict[str, str] = {}  # 源文件名 -> 已保存的目标文件名
    prev_report = download_dir / "ingest_report.csv"
    if prev_report.exists():
        try:
            prev = pd.read_csv(prev_report, dtype=str, encoding="utf-8-sig")
            ok_rows = prev[prev["status"].isin(["ok", "ok_manual", "already_imported"])]
            for _, r in ok_rows.iterrows():
                saved = str(r.get("saved_as") or "").strip()
                # already_imported 是上一轮工具自己的判定，直接信任
                # （其 saved_as 可能为空，历史上不记录）；
                # ok/ok_manual 须目标文件仍在，report 与磁盘不符时允许重导
                if r["status"] == "already_imported" or (
                        saved and (download_dir / saved).exists()):
                    already[r["file"]] = saved
        except Exception:
            pass

    report: list[dict] = []
    files = sorted(p for p in photos_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"{photos_dir} 里没有任何文件")

    for path in files:
        rec = {"file": path.name, "status": "", "point_id": "", "view": "",
               "lat": "", "lon": "", "snap_dist_m": "", "saved_as": "", "message": ""}
        if path.name in already:
            rec.update(status="already_imported", saved_as=already[path.name],
                       message="上一轮已导入，跳过")
            report.append(rec); continue
        ext = path.suffix.lower()

        # ---- 格式关卡 ----
        if ext in HEIC_EXTS:
            rec.update(status="unsupported",
                       message="HEIC 格式需先装 pillow-heif，或在手机上转为 JPG 后重拍/导出")
            report.append(rec); continue
        if ext not in SUPPORTED_EXTS:
            rec.update(status="unsupported",
                       message=f"不支持的格式 {ext or '(无扩展名)'}，支持："
                               f"{'/'.join(sorted(SUPPORTED_EXTS))}")
            report.append(rec); continue

        # ---- EXIF 读取 ----
        try:
            gps = read_exif_gps(path)
        except Exception as e:
            rec.update(status="bad_file",
                       message=f"无法解析图片（文件损坏或不是有效图片）：{type(e).__name__}")
            report.append(rec); continue

        # ---- 坐标来源：EXIF 优先，人工补录兜底 ----
        src = "exif"
        man = manual.get(path.name)
        if gps is None and man:
            if "point_id" not in man and "lat" in man:
                gps = {"lat": man["lat"], "lon": man["lon"], "direction": None}
            src = "manual"
        if gps is None and not (man and "point_id" in man):
            rec.update(status="no_gps",
                       message="照片无 GPS 信息：请在 manual_coords.csv 中补录坐标，"
                               "或打开 ingest_pick_map.html 查 point_id 后填入")
            report.append(rec); continue

        # ---- 归属采样点 ----
        if man and "point_id" in man:
            pid, dist = man["point_id"], 0.0
            valid_pids = set(index.gdf["point_id"])
            if pid not in valid_pids:
                # 宽松匹配：选点地图上显示的是 point_16，用户常只填数字 16
                cand = f"point_{pid}" if pid.isdigit() else None
                if cand and cand in valid_pids:
                    pid = man["point_id"] = cand
                else:
                    example = ", ".join(sorted(valid_pids)[:3])
                    rec.update(status="unmatched",
                               message=f"补录的 point_id={pid} 不存在于采样点中"
                                       f"（格式应为 point_16 这类，如：{example}…）")
                    report.append(rec); continue
        else:
            hit = index.nearest(gps["lat"], gps["lon"])
            pid, dist = hit
            rec["lat"], rec["lon"] = round(gps["lat"], 7), round(gps["lon"], 7)
            if dist > snap_max_m:
                rec.update(status="unmatched",
                           message=f"最近采样点 {pid} 距离 {dist:.0f}m，超过阈值 {snap_max_m:.0f}m；"
                                   f"确认归属可在补录表中直接填 point_id")
                report.append(rec); continue

        # ---- 视角分配 ----
        used = taken.setdefault(pid, set())
        view = classify_view(gps.get("direction") if gps else None,
                             index.bearings.get(pid, 0.0), used)
        if view is None:
            rec.update(status="duplicate", point_id=pid,
                       message=f"{pid} 的 4 个视角已满，本张未导入（如需替换请先删旧图）")
            report.append(rec); continue

        # ---- 归一化保存（统一转 JPG/RGB）----
        out_name = f"{pid}_{view}.jpg"
        try:
            with PILImage.open(path) as img:
                img.convert("RGB").save(download_dir / out_name, quality=92)
        except Exception as e:
            rec.update(status="bad_file", point_id=pid,
                       message=f"转存 JPG 失败：{type(e).__name__}")
            report.append(rec); continue

        used.add(view)
        rec.update(status="ok_manual" if src == "manual" else "ok",
                   point_id=pid, view=view,
                   snap_dist_m=round(dist, 1), saved_as=out_name)
        report.append(rec)

    df = pd.DataFrame(report)
    df.to_csv(download_dir / "ingest_report.csv", index=False, encoding="utf-8-sig")

    # ---- 无 GPS 照片：生成补录模板 + 选点地图 ----
    no_gps = df[df["status"] == "no_gps"]["file"].tolist()
    if no_gps:
        tpl = download_dir / "manual_coords.csv"
        _write_manual_template(tpl, no_gps)
        _build_pick_map(index, download_dir / "ingest_pick_map.html", tianditu_tk)
        print(f"\n{len(no_gps)} 张照片缺 GPS：已生成补录模板 {tpl.name} 和选点地图 "
              f"ingest_pick_map.html，补录后用 --manual manual_coords.csv 重跑")

    # ---- 汇总 ----
    counts = df["status"].value_counts().to_dict()
    n_ok = counts.get("ok", 0) + counts.get("ok_manual", 0)
    print(f"\n导入完成：成功 {n_ok}（含补录 {counts.get('ok_manual', 0)}）| "
          f"跳过已导入 {counts.get('already_imported', 0)} | "
          f"无GPS {counts.get('no_gps', 0)} | 超距 {counts.get('unmatched', 0)} | "
          f"重复 {counts.get('duplicate', 0)} | 无法解析 {counts.get('bad_file', 0)} | "
          f"格式不支持 {counts.get('unsupported', 0)}")
    print(f"报告：{download_dir / 'ingest_report.csv'}")
    # 未匹配的逐张给原因，不用翻报告
    for _, r in df[df["status"] == "unmatched"].iterrows():
        print(f"  [未匹配] {r['file']}: {r['message']}")
    return df
