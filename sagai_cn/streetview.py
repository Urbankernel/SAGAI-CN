"""Block 2 国产化：百度全景静态图批量下载器。

相对上游 Google 版本的关键设计变更（原因见改造方案）：
  1. 百度无免费 Metadata API -> 取消"先查询后下载"，改为直接下载 + 判空。
     百度对无覆盖点返回均色占位图，命中单像素色占比判空规则。
  2. 拿不到 pano_id -> 用影像内容哈希去重替代 pano_id 去重。
  3. 百度不返回真实相机坐标 -> 不做吸附回写，点位保持 Block 1 原始位置，
     同时规避 BD-09 坐标混入成果库的问题。
  4. 配额有限 -> 支持多 AK 轮换 + 请求限速 + 断点续下。

坐标纪律：请求统一带 coordtype=wgs84ll，输入输出全程 WGS84。
"""
from __future__ import annotations

import hashlib
import math
import time
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image as PILImage
from shapely.ops import substring

VIEW_SUFFIXES = ("left", "right", "front", "back")

# 百度错误码中属于"配置类错误"的：换 AK、重试都没用，必须立即终止并给出指引
_FATAL_STATUS = {
    101: "AK 参数不存在或错误：请核对 .env 中的 BAIDU_AK 是否完整、无多余空格",
    102: "MCODE 安全码校验失败：检查应用的安全码（SN）设置，或创建应用时不设校验",
    200: "APP 不存在：AK 无效，请到控制台核对",
    201: "APP 已被禁用：到百度开放平台控制台启用该应用",
    202: "APP 已被管理员删除：请重新创建应用",
    210: "APP IP 白名单校验失败：到应用设置里把 IP 白名单清空或加入本机 IP",
    211: "APP SN 校验失败：服务端应用不要填 SN 校验，或在应用设置中关闭",
    220: "Referer 白名单校验失败：服务端应用不应设置 Referer 白名单",
    230: "APP MCODE 校验失败：检查应用类型与安全码设置",
    240: "APP 服务被禁用：该应用没有开通『全景静态图』服务。"
         "到百度开放平台控制台 -> 应用管理 -> 我的应用 -> 设置，"
         "在『选择服务』中勾选全景静态图 API；若列表里没有，"
         "需先在控制台的服务列表中申请开通该服务",
    250: "用户不存在：AK 对应的账号状态异常",
    260: "密钥错误：AK 复制不完整或已重置，请到控制台重新获取",
}


class BaiduConfigError(RuntimeError):
    """百度侧配置错误（AK/应用/服务未开通），继续跑只会全部失败。"""


# ------------------------------------------------------------------
# 街道走向（与上游 Block 2 一致：canonical bearing 归一化到 [0,180)）
# ------------------------------------------------------------------
def canonical_bearing(raw_bearing: float) -> float:
    """归一化到 [0,180)，保证 left/right 与 OSM 数字化方向无关。"""
    b = raw_bearing % 360
    return b - 180 if b >= 180 else b


def subdivide_segment(line_geom, max_length: float = 50.0) -> list:
    """把线段几何切成 ≤max_length 的 LineString 子段列表。

    本地矢量路网常含 MultiLineString（CAD/DWG 转换、GIS 合并产生），
    而 shapely.ops.substring 只接受 LineString，故先展平为单部件再细分。
    """
    parts = (list(line_geom.geoms)
             if line_geom.geom_type == "MultiLineString" else [line_geom])
    subs = []
    for part in parts:
        total = part.length
        if total <= max_length:
            subs.append(part)
            continue
        n_parts = math.ceil(total / max_length)
        part_length = total / n_parts
        for i in range(n_parts):
            sub = substring(part, i * part_length, min((i + 1) * part_length, total))
            if sub is not None and not sub.is_empty and sub.length > 0:
                subs.append(sub)
    return subs or [line_geom]


def _node_to_node_bearing(line_geom) -> float:
    coords = list(line_geom.coords)
    if len(coords) < 2:
        return 0.0
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return math.degrees(math.atan2(dx, dy)) % 360


def compute_bearing_for_point(point_geom, street_geom, max_sub_length: float = 50.0) -> float:
    sub_segs = subdivide_segment(street_geom, max_sub_length)
    best_seg = min(sub_segs, key=lambda s: s.distance(point_geom))
    return canonical_bearing(_node_to_node_bearing(best_seg))


def get_headings_for_mode(mode: str, bearing: float = 0.0) -> list[tuple[float, str]]:
    """bearing 为 canonical [0,180)。left/right 为垂直街道方向，front/back 沿街道。"""
    front = bearing % 360
    back = (bearing + 180) % 360
    left = (bearing + 270) % 360
    right = (bearing + 90) % 360
    if mode == "left_right":
        return [(left, "left"), (right, "right")]
    if mode == "front_back":
        return [(front, "front"), (back, "back")]
    return [(left, "left"), (right, "right"), (front, "front"), (back, "back")]


# ------------------------------------------------------------------
# 百度全景下载
# ------------------------------------------------------------------
class BaiduPanoClient:
    """百度全景静态图 API 客户端：多 AK 轮换 + 限速 + 判空。"""

    def __init__(
        self,
        aks: list[str],
        base_url: str = "https://api.map.baidu.com/panorama/v2",
        width: int = 640,
        height: int = 640,
        request_interval: float = 0.2,
        timeout: int = 15,
    ):
        if not aks:
            raise ValueError("未配置百度 AK：设置环境变量 BAIDU_AK（多个用英文逗号分隔）")
        self.aks = aks
        self.base_url = base_url
        self.width = width
        self.height = height
        self.request_interval = request_interval
        self.timeout = timeout
        self._ak_index = 0
        self._last_request_ts = 0.0

    def _next_ak(self) -> str:
        ak = self.aks[self._ak_index % len(self.aks)]
        self._ak_index += 1
        return ak

    def _throttle(self) -> None:
        wait = self.request_interval - (time.time() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.time()

    def fetch(self, lat: float, lon: float, heading: float, pitch: int, fov: int,
              max_retries: int = 3) -> PILImage.Image | None:
        """下载一张全景切片。无覆盖/失败返回 None。

        注意百度 location 参数为 经度在前（与 Google 相反）。
        """
        params = (
            f"location={lon},{lat}&coordtype=wgs84ll"
            f"&width={self.width}&height={self.height}"
            f"&heading={heading:.1f}&pitch={pitch}&fov={fov}"
        )
        last_err: Exception | None = None
        for attempt in range(max_retries):
            ak = self._next_ak()
            url = f"{self.base_url}?ak={ak}&{params}"
            try:
                self._throttle()
                resp = requests.get(url, timeout=self.timeout)
                ctype = resp.headers.get("Content-Type", "")
                if resp.status_code == 200 and "image" in ctype:
                    return PILImage.open(BytesIO(resp.content)).convert("RGB")
                # 非图片响应：解析百度错误码，区分配置错误（立即终止）与配额/临时错误（换 AK 重试）
                resp.encoding = "utf-8"  # 修正乱码，让错误信息可读
                try:
                    err = resp.json()
                    status = int(err.get("status", -1))
                    message = err.get("message", "")
                except Exception:
                    status, message = -1, resp.text[:200]
                if status in _FATAL_STATUS:
                    raise BaiduConfigError(
                        f"百度接口配置错误（status={status} {message}）：\n"
                        f"  {_FATAL_STATUS[status]}\n"
                        f"  修复后重新运行同一指令即可（断点续跑，已下载的不会重来）")
                if status in (301, 302):
                    last_err = RuntimeError(
                        f"配额超限（status={status}）：换 AK 重试；全部 AK 用尽请明天再跑（断点续跑）")
                else:
                    last_err = RuntimeError(
                        f"HTTP {resp.status_code} status={status}: {message}")
            except BaiduConfigError:
                raise
            except Exception as e:  # 网络异常
                last_err = e
            time.sleep(min(2 ** attempt, 8))
        print(f"  [warn] 下载失败 lat={lat},lon={lon},heading={heading}: {last_err}")
        return None


def is_blank_image(image: PILImage.Image, threshold: float = 0.9) -> bool:
    """单像素色占比 > threshold 判为空图/占位图（沿用上游规则）。

    百度"暂无全景"占位图为均色灰图，天然命中此规则。
    """
    arr = np.array(image)
    _, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    return counts.max() / counts.sum() > threshold


# ------------------------------------------------------------------
# 主流程（对应上游 run_block2，Phase 2 按百度能力重设计）
# ------------------------------------------------------------------
def download_streetview(
    gpkg_path: Path,
    download_dir: Path,
    client: BaiduPanoClient,
    mode: str = "4_views",
    pitch: int = -20,
    fov: int = 90,
    max_sub_m: float = 50.0,
    dedup_by_hash: bool = True,
) -> pd.DataFrame:
    """批量下载。返回下载报告 DataFrame（也写入 download_report.csv）。

    断点续下：已存在的 {point_id}_{view}.jpg / _NA.jpg 直接跳过。
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    points = gpd.read_file(gpkg_path, layer="points")
    streets = gpd.read_file(gpkg_path, layer="streets")

    # Phase 1: 在原始点位上算 bearing（点位不吸附，顺序天然安全）
    proj_crs = points.estimate_utm_crs()
    points_proj = points.to_crs(proj_crs)
    streets_proj = streets.to_crs(proj_crs)
    street_lookup = dict(zip(streets_proj["street_id"], streets_proj.geometry))

    bearings: dict[str, float] = {}
    for _, row in points_proj.iterrows():
        geom = street_lookup.get(row.get("street_id"))
        bearings[row["point_id"]] = (
            round(compute_bearing_for_point(row.geometry, geom, max_sub_m), 1)
            if geom is not None else 0.0
        )

    # bearing 回写 gpkg（供成果追溯；坐标本身不动，仍是原始 WGS84 点位）
    points["bearing"] = points["point_id"].map(bearings)
    points.to_file(gpkg_path, layer="points", driver="GPKG")

    # Phase 2: 逐点逐视角下载（替代原 metadata 过滤阶段）
    report: list[dict] = []
    n_new = n_skip = n_blank = n_fail = 0
    seen_hashes: dict[str, str] = {}  # md5 -> 首个文件名（内容去重）

    for i, (_, row) in enumerate(points.iterrows()):
        pid = row["point_id"]
        lat, lon = row["Latitude"], row["Longitude"]
        headings = get_headings_for_mode(mode, bearings.get(pid, 0.0))

        for heading, suffix in headings:
            base = f"{pid}_{suffix}"
            expected = [download_dir / f"{base}.jpg", download_dir / f"{base}_NA.jpg"]
            if any(p.exists() for p in expected):
                n_skip += 1
                continue

            img = client.fetch(lat, lon, heading, pitch, fov)
            if img is None:
                n_fail += 1
                report.append({"image_name": f"{base}.jpg", "status": "download_failed"})
                continue

            digest = hashlib.md5(img.tobytes()).hexdigest()
            status = "ok"
            if is_blank_image(img):
                status = "blank"
                base += "_NA"
                n_blank += 1
            elif dedup_by_hash and digest in seen_hashes:
                status = f"dup_of:{seen_hashes[digest]}"
                base += "_NA"  # 重复影像按无效处理，不参与打分
                n_blank += 1
            else:
                seen_hashes[digest] = f"{base}.jpg"
                n_new += 1

            img.save(download_dir / f"{base}.jpg", quality=90)
            report.append({"image_name": f"{base}.jpg", "status": status,
                           "lat": lat, "lon": lon, "heading": round(heading, 1)})

        if (i + 1) % 20 == 0 or (i + 1) == len(points):
            print(f"  进度 {i + 1}/{len(points)} 点 | 新 {n_new} 空/重 {n_blank} "
                  f"跳过 {n_skip} 失败 {n_fail}")

    report_df = pd.DataFrame(report)
    report_df.to_csv(download_dir / "download_report.csv", index=False, encoding="utf-8-sig")

    n_with = len({
        p.name.rsplit("_", 1)[0]
        for p in download_dir.glob("point_*.jpg") if "_NA" not in p.name
    })
    coverage = n_with / max(len(points), 1)
    print(f"下载完成：新图 {n_new}，空/重复 {n_blank}，续跑跳过 {n_skip}，失败 {n_fail}")
    print(f"点位覆盖率：{n_with}/{len(points)} = {coverage:.1%}")
    if coverage < 0.5:
        print("[warn] 覆盖率低于 50%，该区域百度全景覆盖不足，结果慎用")
    return report_df
