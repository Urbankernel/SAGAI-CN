"""Block 2 国产化：百度街景全景逆向下载 + 全景切四视角。

替代官方「全景静态图 API」——个人开发者无调用权限，且官方版每点要发
4 次请求、只给 640×640 低清方向图。本模块走逆向接口（mapsv0.bdimg.com
的 qt=qsdata/sdata/pdata，无需 AK）：

    WGS84 经纬度
      → BD09MC 百度墨卡托坐标
      → qsdata 拿 panoid（全景唯一标识）
      → pdata 逐块下载瓦片
      → Pillow 拼接为等距柱状投影全景图（8192×4096 级别可调）
      → 按采样点 bearing 切成 left/right/front/back 四张方向图
      → 命名 {point_id}_{view}.jpg，与官方版/自采照片路线一致，
        下游 infer/aggregate/map 零改动。

坐标纪律：WGS84 → GCJ-02 → BD-09 → BD09MC 转换链仅用于发请求（逆向接口
要求 BD09MC 墨卡托坐标），成果点位仍是 gpkg 的 WGS84，与 streetview.py 一致。

参考（已跑通的逆向脚本）：F:/城市研究领域Agent/baidu_streetview.py
"""
from __future__ import annotations

import math
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image

from .streetview import (VIEW_SUFFIXES, compute_bearing_for_point,
                         get_headings_for_mode)

# ----------------------------------------------------------------------------
# 坐标系转换：WGS-84 -> GCJ-02 -> BD-09 -> BD09MC（移植自逆向脚本）
# ----------------------------------------------------------------------------

# 百度墨卡托（BD09MC）分段多项式系数（百度地图 JS API 官方算法）。
# 注意：BD09MC 不是标准 Web Mercator！用标准公式纬度方向可偏差 20+ 公里，
# 街景稀疏区会误判为"无街景"，必须用下面的分段多项式。
LLBAND = [75, 60, 45, 30, 15, 0]
LL2MC = [
    [-0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340, 26112667856603880, -35149669176653700, 26595700718403920, -10725012454188240, 1800819912950474, 82.5],
    [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142, -15171875531.51559, 12053065933.6625, -5124939663.577472, 913311935.9512032, 67.5],
    [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455, -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5],
    [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984122, 144416.9293806241, 37.5],
    [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.735728697, 6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746631, 1405.483844121726, 22.5],
    [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078884, 0.37238884252424, 7.45],
]


def _out_of_china(lng, lat):
    return not (73.66 <= lng <= 135.05 and 3.86 <= lat <= 53.55)


def _transform_lat(lng, lat):
    ret = (-100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat
           + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng)))
    ret += (20.0 * math.sin(6.0 * lng * math.pi)
            + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi)
            + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi)
            + 320.0 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng, lat):
    ret = (300.0 + lng + 2.0 * lat + 0.1 * lng * lng
           + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng)))
    ret += (20.0 * math.sin(6.0 * lng * math.pi)
            + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi)
            + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi)
            + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng, lat):
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - 0.00669342162296594323 * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323))
                             / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic
                             * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat


def gcj02_to_bd09(lng, lat):
    x_pi = math.pi * 3000.0 / 180.0
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    return (z * math.cos(theta) + 0.0065,
            z * math.sin(theta) + 0.006)


def bd09_to_bd09mc(lng, lat):
    lat_abs = abs(lat)
    coef = LL2MC[-1]
    for i in range(len(LLBAND)):
        if lat_abs >= LLBAND[i]:
            coef = LL2MC[i]
            break
    x = coef[0] + coef[1] * abs(lng)
    cc = lat_abs / coef[9]
    y = coef[2]
    for j in range(3, 9):
        y += coef[j] * cc ** (j - 2)
    x = -x if lng < 0 else x
    y = -y if lat < 0 else y
    return x, y


def wgs84_to_bd09mc(lng, lat):
    lng, lat = wgs84_to_gcj02(lng, lat)
    lng, lat = gcj02_to_bd09(lng, lat)
    return bd09_to_bd09mc(lng, lat)


# ----------------------------------------------------------------------------
# 逆向接口下载
# ----------------------------------------------------------------------------

# 各清晰度级别瓦片布局（ImgLevel -> (列, 行, 请求时 z 值)）
LEVEL_MAP = {
    1: (2, 1, 2),
    2: (4, 2, 3),
    3: (8, 4, 4),
    4: (16, 8, 5),
}
TILE_SIZE = 512

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Referer": "https://map.baidu.com/",
}


class BaiduReverseClient:
    """逆向接口客户端：qsdata 拿 panoid + pdata 逐瓦片下载拼接全景。"""

    def __init__(self, sleep_seconds: float = 0.3, timeout: int = 30,
                 max_retries: int = 2):
        self.session = requests.Session()
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, url: str):
        for attempt in range(self.max_retries + 1):
            try:
                return self.session.get(url, headers=HEADERS,
                                        timeout=self.timeout)
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(self.sleep_seconds)
                    continue
                raise

    def get_panoid(self, mc_x: float, mc_y: float):
        """qt=qsdata：用 BD09MC 墨卡托坐标取 panoid。无街景返回 None。

        错误分类：``error==0`` 成功；``error==404`` 该坐标无街景（不重试）；
        其他 error（限流/服务端抖动）重试 ``max_retries`` 次再判定——批量下载
        时偶发撞上 error 会被误判为"无街景"，重试可显著降低误判。
        """
        url = "https://mapsv0.bdimg.com/?qt=qsdata&x=%.6f&y=%.6f" % (mc_x, mc_y)
        for attempt in range(self.max_retries + 1):
            data = self._get(url).json()
            err = data.get("result", {}).get("error")
            if err == 0:
                return data.get("content", {}).get("id")
            if err == 404:
                return None  # 真无街景，无需重试
            if attempt < self.max_retries:  # 限流/抖动：稍候重试
                time.sleep(self.sleep_seconds)
        return None

    def get_camera_info(self, panoid: str) -> dict | None:
        """qt=sdata：取采集车拍摄元数据（Heading/Pitch/拍摄日期等）。失败返回 None。

        其中 ``Heading`` 是采集车拍摄时的朝向（0=正北，顺时针，度），是全景图
        方向校准的关键真值：百度约定全景图第 0 列 = Heading 的左侧（即绝对方位
        Heading-90°），所以切片时需要逐点用 (90 - Heading) 修正，见 slice_view。

        与 get_panoid 一致地重试：``error != 0``（限流/服务端抖动）与网络异常都
        重试 ``max_retries`` 次。批量下载时 sdata 偶发抖动（实测约 2.6%），不重试
        会把个别点误判为"无 Heading"、按旧约定错切方向。
        """
        url = "https://mapsv0.bdimg.com/?qt=sdata&sid=%s" % panoid
        for attempt in range(self.max_retries + 1):
            try:
                data = self._get(url).json()
            except Exception:
                data = None
            if data is not None and data.get("result", {}).get("error") == 0:
                content = data.get("content")
                # 结构正常但无内容：panoid 存在却查不到元数据，重试无意义
                return content[0] if isinstance(content, list) and content else None
            if attempt < self.max_retries:
                time.sleep(self.sleep_seconds)
        return None

    def find_nearest_panoid(self, mc_x: float, mc_y: float,
                            max_radius: float = 5.0):
        """在坐标周围环形搜索最近的街景点，返回 (panoid, 距离米) 或 (None, None)。

        修正自 baidu_streetview.py 的同名函数：其原版 ``radius`` 从 100 起步、
        步进 150、上限却只有 50，循环体实际永不执行（bug）。此处从近到远分档
        （max_radius 的 40%/70%/100%）采样 8 个方向，避免固定档位在 max_radius
        较小时（如 5 米）被全部 break 掉导致搜索失效。BD09MC 在几十米范围内
        近似等距，直接用米做偏移。
        """
        for frac in (0.4, 0.7, 1.0):
            radius = max_radius * frac
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                xx = mc_x + radius * math.cos(rad)
                yy = mc_y + radius * math.sin(rad)
                pid = self.get_panoid(xx, yy)
                if pid:
                    return pid, radius
                time.sleep(self.sleep_seconds)
        return None, None

    def download_tile(self, panoid: str, row: int, col: int, z: int) -> Image.Image:
        """qt=pdata：下载单块瓦片。"""
        url = ("https://mapsv0.bdimg.com/?qt=pdata&sid=%s&pos=%d_%d&z=%d"
               % (panoid, row, col, z))
        resp = self._get(url)
        return Image.open(BytesIO(resp.content))

    def download_panorama(self, panoid: str, level: int = 3,
                          progress_cb=None) -> Image.Image:
        """下载并拼接指定清晰度级别的完整全景图（等距柱状投影）。

        progress_cb(done, total)：每下载完一块瓦片回调一次，用于终端进度显示；
        传 None 则静默。
        """
        cols, rows, z = LEVEL_MAP[level]
        total = cols * rows
        pano = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
        done = 0
        for row in range(rows):
            for col in range(cols):
                tile = self.download_tile(panoid, row, col, z)
                pano.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total)
                time.sleep(self.sleep_seconds)
        return pano


# ----------------------------------------------------------------------------
# 全景裁剪 + 切四视角
# ----------------------------------------------------------------------------

# 模拟官方静态图 API 的取景参数：相机俯仰 -20°、视场角 90°
PITCH_CENTER = -20.0
FOV_H = 90.0
FOV_V = 90.0
# 全景图第 0 列的方向约定（2026-10-17 真实抓图实测校准，推翻了旧假设
# "第 0 列 = 正北"）：百度全景图第 0 列 = 采集车朝向(Heading)的左侧，即绝对方位
# (Heading - 90°)。因此切方位角 h 的视角，实际列中心 = (h - Heading + 90)/360*W，
# 等价于给 h 加偏移 (90 - Heading)。该偏移随每个 panoid 的采集车朝向变化，
# 不是全局常量——reverse 流程对每个 panoid 调 qt=sdata 取 Heading 后传入
# slice_views(..., heading_offset=90 - Heading)；Heading 缺失时退化为 0 并告警。


def crop_streetview(img: Image.Image, bottom_ratio: float = 0.0,
                    top_ratio: float = 0.0) -> Image.Image:
    """按比例裁剪等距柱状投影全景图（照搬 baidu_streetview.py）。

    bottom_ratio：裁掉底部比例（0.4 = 去掉下方 40% 的采集车顶/车头区域）
    top_ratio   ：裁掉顶部比例（0.2 = 去掉上方 20% 的天空）
    两项均设为 0 可关闭裁剪，保留完整全景。
    """
    bottom_ratio = max(0.0, min(bottom_ratio, 0.9))
    top_ratio = max(0.0, min(top_ratio, 0.9))
    if top_ratio + bottom_ratio >= 1.0:
        raise ValueError("crop_top + crop_bottom 之和必须小于 1")
    w, h = img.size
    top = int(round(h * top_ratio))
    bottom = int(round(h * (1.0 - bottom_ratio)))
    if bottom - top < 1:
        return img
    return img.crop((0, top, w, bottom))


def slice_view(pano: Image.Image, heading: float,
               crop_bottom: float = 0.0, crop_top: float = 0.0,
               heading_offset: float = 0.0) -> Image.Image:
    """从等距柱状全景图切出一个方向的视角窗口（水平 fov 90°、俯仰 -65°~+25°）。

    等距柱状投影：宽 W 对应 360° 水平方位，高 H 对应 180° 垂直（顶部 +90°）。
    ``heading`` 为方位角（0=北，顺时针），窗口跨 0°/360° 边界时自动环绕。

    ``heading_offset``：全景图第 0 列并非正北（实测第 0 列 = 采集车朝向的左侧），
    传入 (90 - 采集车Heading) 做逐点修正；详见模块注释。0 表示不修正。

    crop_bottom / crop_top：先按比例裁掉底部车顶 / 顶部天空（等价于先调
    crop_streetview 再切片），取景窗口被裁剪范围约束——例如 crop_bottom=0.4
    时，窗口底部从 -65° 抬到 -18°，去掉更多采集车车顶区域。
    """
    W, H = pano.size
    h = (heading + heading_offset) % 360
    cx = h / 360.0 * W                       # 窗口中心列
    half = FOV_H / 360.0 * W / 2.0           # 半宽（像素）
    width = int(round(half * 2))
    x0 = cx - half
    # 垂直窗口：先按 crop 裁掉车顶/天空，再取 pitch [-65°, +25°]，并约束在裁剪范围内
    crop_bottom = max(0.0, min(crop_bottom, 0.9))
    crop_top = max(0.0, min(crop_top, 0.9))
    pitch_bottom = -90.0 + crop_bottom * 180.0      # 裁剪后最低俯仰
    pitch_top = 90.0 - crop_top * 180.0             # 裁剪后最高俯仰
    win_top = min(PITCH_CENTER + FOV_V / 2.0, pitch_top)        # 窗口上边界（+25° 或更低）
    win_bottom = max(PITCH_CENTER - FOV_V / 2.0, pitch_bottom)  # 窗口下边界（-65° 或更高）
    y0 = int(round((90.0 - win_top) / 180.0 * H))
    y1 = int(round((90.0 - win_bottom) / 180.0 * H))
    y1 = max(y1, y0 + 1)                     # 极端 crop 下保证非空
    # 水平滚动对齐到窗口起点，天然处理跨 0°/360° 环绕
    arr = np.asarray(pano)
    rolled = np.roll(arr, -int(round(x0)), axis=1)
    return Image.fromarray(rolled[y0:y1, 0:width, :])


def slice_views(pano: Image.Image, bearing: float,
                crop_bottom: float = 0.0, crop_top: float = 0.0,
                heading_offset: float = 0.0) -> dict[str, Image.Image]:
    """按采样点 bearing 把全景切成四视角。返回 {suffix: Image}。

    heading_offset：逐点方向修正（见 slice_view），由调用方按 sdata 的 Heading 计算。
    """
    return {suffix: slice_view(pano, heading, crop_bottom, crop_top, heading_offset)
            for heading, suffix in get_headings_for_mode("4_views", bearing)}


# ----------------------------------------------------------------------------
# 图片文件命名：{point_id}_{路名}_{拍摄时间}_{suffix}.jpg
# ----------------------------------------------------------------------------
# 文件系统非法字符（Windows 文件名不允许 \ / : * ? " < > |，连同空白一并去掉）
_ILLEGAL_FS = re.compile(r'[\\/:*?"<>|\s]+')
_DATE8 = re.compile(r"^\d{8}$")


def _safe_road_name(name) -> str:
    """把 sdata 的路名/日期清洗成可入文件名的安全串；空/无则返回 ''。"""
    if not name:
        return ""
    return _ILLEGAL_FS.sub("", str(name))


def _img_stem(pid: str, road_name, shot_date, suffix: str) -> str:
    """图片文件名主干：{pid}_{路名}_{拍摄时间}_{suffix}。

    路名、拍摄时间均为可选段，缺失则省略对应段；两者都缺时退化为 {pid}_{suffix}，
    与旧命名契约一致，下游 infer/aggregate 不受影响。
    """
    meta = "_".join(
        x for x in (_safe_road_name(road_name), _safe_road_name(shot_date)) if x)
    return f"{pid}_{meta}_{suffix}" if meta else f"{pid}_{suffix}"


def _has_image(download_dir: Path, pid: str, suffix: str) -> bool:
    """该点某方向图片是否存在（兼容带/不带路名两种命名）。"""
    if list(download_dir.glob(f"{pid}_*_{suffix}.jpg")):
        return True
    return (download_dir / f"{pid}_{suffix}.jpg").exists()


def _parse_road_date(download_dir: Path, pid: str, suffixes) -> tuple[str | None, str | None]:
    """从已下载文件名解析路名与拍摄日期（断点续跑恢复报告信息用）。

    文件名格式 {pid}_{路名}_{日期}_{suffix}.jpg（可选段可缺失）。解析规则：中间段
    以 "_" 拆分，最后一段为 8 位纯数字时视为拍摄日期、其余拼接为路名。
    旧命名 {pid}_{suffix}.jpg 无中间段，返回 (None, None)。
    """
    for s in suffixes:
        cand = sorted(download_dir.glob(f"{pid}_*_{s}.jpg"))
        legacy = download_dir / f"{pid}_{s}.jpg"
        path = cand[0] if cand else (legacy if legacy.exists() else None)
        if path is None:
            continue
        stem = path.stem
        if stem == f"{pid}_{s}":  # 旧命名，无路名/日期段
            return None, None
        meta = stem[len(pid) + 1:]
        if meta.endswith("_" + s):
            meta = meta[: -len(s) - 1]
        parts = meta.split("_") if meta else []
        if not parts:
            return None, None
        if _DATE8.match(parts[-1]):
            date = parts[-1]
            road = "_".join(parts[:-1]) or None
        else:
            date, road = None, "_".join(parts) or None
        return road, date
    return None, None


def _covered_point_ids(download_dir: Path, suffixes, point_ids) -> set[str]:
    """统计有产出影像的点集（含断点续跑此前下载的），按 point_ids 逐个查文件。"""
    return {
        pid for pid in point_ids
        if any(_has_image(download_dir, pid, s) for s in suffixes)
    }


# ----------------------------------------------------------------------------
# 终端进度条（零依赖，纯 ASCII 保证 Windows 控制台不乱码）
# ----------------------------------------------------------------------------

def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _render_bar(done: int, total: int, width: int = 24) -> str:
    """渲染 [####----] 形式进度条 + 计数 + 百分比。"""
    frac = min(done / max(total, 1), 1.0)
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} {frac * 100:5.1f}%"


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def download_streetview_reverse(
    gpkg_path: Path,
    download_dir: Path,
    level: int = 3,
    sleep_seconds: float = 0.3,
    timeout: int = 30,
    max_retries: int = 2,
    max_sub_m: float = 50.0,
    crop_bottom: float = 0.4,
    crop_top: float = 0.0,
    search_radius: float = 5.0,
) -> pd.DataFrame:
    """逆向接口批量下载街景并切四视角。返回下载报告（也写 download_report.csv）。

    图片命名 {point_id}_{路名}_{拍摄时间}_{view}.jpg（路名/时间缺失时省略对应段，
    退化回 {point_id}_{view}.jpg）。断点续下：某点四视角齐全即跳过。无街景点记
    status=no_pano（下次重跑会再试一次 qsdata，单次请求成本极低）。

    crop_bottom / crop_top：全景裁剪比例（裁掉底部采集车顶 / 顶部天空，参照
    baidu_streetview.py），默认 0.4 / 0.0；两项都设为 0 关闭裁剪。

    search_radius：坐标本身无街景时的就近搜索半径（米，默认 5）；设 0 关闭。
    采样点是沿路网等间距插值的，未必精确落在百度街景采集轨迹上，就近搜索可
    找回轨迹附近的街景点（对齐 baidu_streetview.py 的 search_nearest）。
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    points = gpd.read_file(gpkg_path, layer="points")
    streets = gpd.read_file(gpkg_path, layer="streets")
    suffixes = [s for _, s in get_headings_for_mode("4_views", 0.0)]

    # bearing：点位不吸附，街道走向用于切片定向（与官方版 Phase 1 一致）
    proj = points.estimate_utm_crs()
    pts_p = points.to_crs(proj)
    sts_p = streets.to_crs(proj)
    lookup = dict(zip(sts_p["street_id"], sts_p.geometry))
    bearings: dict[str, float] = {}
    for _, row in pts_p.iterrows():
        geom = lookup.get(row.get("street_id"))
        bearings[row["point_id"]] = (
            round(compute_bearing_for_point(row.geometry, geom, max_sub_m), 1)
            if geom is not None else 0.0)
    points["bearing"] = points["point_id"].map(bearings)
    points.to_file(gpkg_path, layer="points", driver="GPKG")

    client = BaiduReverseClient(sleep_seconds, timeout, max_retries)
    report: list[dict] = []
    n_ok = n_skip = n_nopano = n_fail = 0
    total = len(points)
    # 交互终端用 \r 原地刷新单行进度条；重定向到文件/日志时降级为每 20 点打一行
    use_inline = _is_tty(sys.stderr)

    for i, (_, row) in enumerate(points.iterrows()):
        pid = str(row["point_id"])
        lat, lon = row["Latitude"], row["Longitude"]

        if all(_has_image(download_dir, pid, s) for s in suffixes):
            n_skip += 1
            # 跳过的点也写报告：从文件名恢复路名/拍摄日期，否则重跑后
            # download_report.csv 被整体覆盖，先下载点的路名/日期会丢失，
            # 地图弹窗的路名随之缺失（对齐 Baidu_StreetView_Download 的修复）
            road_name, shot_date = _parse_road_date(download_dir, pid, suffixes)
            rep = {"image_name": pid, "status": "ok", "lat": lat, "lon": lon}
            if road_name:
                rep["road_name"] = road_name
            if shot_date:
                rep["shot_date"] = shot_date
            report.append(rep)
        else:
            mc_x, mc_y = wgs84_to_bd09mc(lon, lat)
            panoid = None
            nearest_m = None
            err = None
            try:
                panoid = client.get_panoid(mc_x, mc_y)
            except Exception as e:
                err = e
            # 坐标本身无街景时，就近搜索兜底（对齐 baidu_streetview.py 的
            # search_nearest，修正其原版 radius 永不执行的 bug）
            if err is None and not panoid and search_radius > 0:
                try:
                    panoid, nearest_m = client.find_nearest_panoid(
                        mc_x, mc_y, search_radius)
                except Exception:
                    panoid, nearest_m = None, None
            if err is not None:
                n_fail += 1
                report.append({"image_name": pid, "status": "download_failed",
                               "lat": lat, "lon": lon,
                               "message": f"{type(err).__name__}"})
            elif not panoid:
                n_nopano += 1
                report.append({"image_name": pid, "status": "no_pano",
                               "lat": lat, "lon": lon})
            else:
                # 瓦片级进度回调：主进度始终显示点级（第几个点/共几个点），
                # 瓦片进度作为从属信息跟在同一行，避免 32 块瓦片的子进度
                # 反复覆盖点级进度、让用户误以为总共只有 32 个任务
                def tile_cb(done, total_tiles, _pid=pid, _done_pts=i):
                    if use_inline:
                        sys.stderr.write(
                            "\r" + _render_bar(_done_pts, total)
                            + f" 点 | " + _render_bar(done, total_tiles, width=12)
                            + f" 瓦片（{_pid}）      ")
                        sys.stderr.flush()

                try:
                    # 方向校准 + 路名 + 拍摄时间：先取 sdata（Heading/Rname/Date），
                    # 再下瓦片拼全景。提前到瓦片下载前，既避开 32 块瓦片高频请求后的
                    # 限流窗口，失败也能更快重试（sdata 单次请求成本远低于 32 块瓦片）。
                    car_heading: float | None = None
                    road_name: str | None = None
                    shot_date: str | None = None
                    cam = client.get_camera_info(panoid)
                    if cam:
                        if cam.get("Rname"):
                            road_name = str(cam["Rname"])
                        if cam.get("Date"):
                            shot_date = str(cam["Date"])
                        if "Heading" in cam:
                            car_heading = float(cam["Heading"])
                    if car_heading is None:
                        print(f"  [warn] {pid}: sdata 未返回 Heading，"
                              f"方向图可能整体转偏，已按旧约定切片")
                    pano = client.download_panorama(panoid, level,
                                                    progress_cb=tile_cb)
                    # 全景图第 0 列 = 采集车朝向(Heading)的左侧，
                    # 切方位角 h 需加偏移 (90 - Heading)，见模块注释
                    heading_offset = (
                        (90.0 - car_heading) % 360
                        if car_heading is not None else 0.0)
                    for suffix, img in slice_views(
                            pano, bearings.get(pid, 0.0),
                            crop_bottom, crop_top,
                            heading_offset=heading_offset).items():
                        stem = _img_stem(pid, road_name, shot_date, suffix)
                        img.save(download_dir / f"{stem}.jpg", quality=90)
                    n_ok += 1
                    rep = {"image_name": pid, "status": "ok",
                           "lat": lat, "lon": lon, "panoid": panoid}
                    if road_name:
                        rep["road_name"] = road_name
                    if shot_date:
                        rep["shot_date"] = shot_date
                    if car_heading is not None:
                        rep["car_heading"] = round(car_heading, 2)
                    if nearest_m is not None:
                        rep["message"] = f"就近匹配 {nearest_m:.0f}m"
                    report.append(rep)
                except Exception as e:
                    n_fail += 1
                    report.append({"image_name": pid, "status": "download_failed",
                                   "lat": lat, "lon": lon,
                                   "message": f"{type(e).__name__}"})

        # 点级进度：原地刷新或降级打印（行尾长空格清除瓦片行残留字符）
        if use_inline:
            sys.stderr.write(
                "\r" + _render_bar(i + 1, total) + " 点"
                + f"  新{n_ok} 无街景{n_nopano} 跳过{n_skip} 失败{n_fail}"
                + " " * 24)
            sys.stderr.flush()
        elif (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  进度 {i + 1}/{total} 点 | 新 {n_ok} 无街景 {n_nopano} "
                  f"跳过 {n_skip} 失败 {n_fail}")

    if use_inline:
        sys.stderr.write("\n")
        sys.stderr.flush()

    report_df = pd.DataFrame(report)
    try:
        report_df.to_csv(download_dir / "download_report.csv",
                         index=False, encoding="utf-8-sig")
    except PermissionError:
        print("[warn] download_report.csv 写入失败（文件可能正被 Excel 打开占用）。"
              "本次结果未能记录，但街景图片已正常保存，可关闭占用后重跑以刷新报告。")

    n_with = len(_covered_point_ids(download_dir, suffixes, points["point_id"]))
    coverage = n_with / max(len(points), 1)
    print(f"下载完成：新点 {n_ok}，无街景 {n_nopano}，续跑跳过 {n_skip}，失败 {n_fail}")
    print(f"点位覆盖率：{n_with}/{len(points)} = {coverage:.1%}")
    if coverage < 0.5:
        print("[warn] 覆盖率低于 50%，该区域百度全景覆盖不足，结果慎用")
    return report_df
