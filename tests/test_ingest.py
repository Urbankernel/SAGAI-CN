#!/usr/bin/env python
"""ingest 模块端到端验证（不依赖真实照片/网络）。

用 piexif 合成带 GPS 的测试照片，覆盖：
  ok（EXIF 吸附）、方向归类视角、无 GPS→补录（坐标与 point_id 两种方式）、
  超距 unmatched、视角占满 duplicate、坏文件 bad_file、
  不支持格式 unsupported、断点续导不重复导入。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import piexif
from PIL import Image

from sagai_cn.ingest import ingest_photos, read_exif_gps, classify_view
from sagai_cn.sampling import generate_points, save_case_gpkg
import geopandas as gpd
from shapely.geometry import LineString, box

PASS = []
def check(name, cond):
    PASS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def make_photo(path: Path, lat=None, lon=None, direction=None):
    img = Image.new("RGB", (64, 64), (120, 80, 40))
    if lat is None:
        img.save(path)  # 无 EXIF
        return
    def dms(v):
        v = abs(v)
        d, m = int(v), int((v - int(v)) * 60)
        s = round((v - d - m / 60) * 3600 * 100)
        return ((d, 1), (m, 1), (s, 100))
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: dms(lon),
    }
    if direction is not None:
        gps[piexif.GPSIFD.GPSImgDirection] = (int(direction * 100), 100)
    exif_bytes = piexif.dump({"GPS": gps})
    img.save(path, exif=exif_bytes)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ingest_test_"))
    print(f"临时目录：{tmp}")

    # 合成路网与采样点（横街东西向，bearing≈90）
    streets = gpd.GeoDataFrame(
        {"highway": ["residential"], "name": ["测试街"], "osmid": [1]},
        geometry=[LineString([(120.0, 36.0), (120.01, 36.0)])], crs="EPSG:4326")
    points, streets = generate_points(box(119.999, 35.999, 120.011, 36.001),
                                      streets, spacing_m=200, offset_m=100)
    gpkg = tmp / "t_osm.gpkg"
    save_case_gpkg(points, streets, gpkg)
    p0 = points.iloc[0]
    lat0, lon0 = p0["Latitude"], p0["Longitude"]
    pid0 = p0["point_id"]
    print(f"采样点 {pid0}: ({lat0:.6f}, {lon0:.6f}), 共 {len(points)} 点")

    photos = tmp / "photos"
    photos.mkdir()
    # 1. 正常：贴近 point_1，带方向 90°（≈front，因街道 bearing≈90）
    make_photo(photos / "a_ok.jpg", lat0 + 0.00001, lon0, direction=90)
    # 2. 正常：同点，方向 0°（最接近 left=0，因 left=(90+270)%360=0）
    make_photo(photos / "b_ok.jpg", lat0, lon0 + 0.00001, direction=0)
    # 3. 无方向信息 → 按空位分配
    make_photo(photos / "c_nodir.jpg", lat0, lon0)
    # 4. 无 GPS → no_gps
    make_photo(photos / "d_nogps.jpg")
    # 5. 超距（离最近点 >500m）→ unmatched
    make_photo(photos / "e_far.jpg", lat0 + 0.01, lon0)
    # 6. 坏文件 → bad_file
    (photos / "f_bad.jpg").write_bytes(b"not an image at all")
    # 7. 不支持格式 → unsupported
    (photos / "g_doc.txt").write_text("hello")
    (photos / "h_heic.heic").write_bytes(b"fake")

    out = tmp / "images"
    df = ingest_photos(gpkg, photos, out, snap_max_m=50)
    st = dict(zip(df["file"], df["status"]))

    print("== 第一轮导入 ==")
    check("a_ok 吸附成功", st["a_ok.jpg"] == "ok")
    check("b_ok 吸附成功", st["b_ok.jpg"] == "ok")
    check("c_nodir 吸附成功", st["c_nodir.jpg"] == "ok")
    check("d_nogps 标记 no_gps", st["d_nogps.jpg"] == "no_gps")
    check("e_far 标记 unmatched", st["e_far.jpg"] == "unmatched")
    check("f_bad 标记 bad_file", st["f_bad.jpg"] == "bad_file")
    check("g_doc 标记 unsupported", st["g_doc.txt"] == "unsupported")
    check("h_heic 标记 unsupported 且提示 HEIC",
          st["h_heic.heic"] == "unsupported"
          and "HEIC" in df.loc[df.file == "h_heic.heic", "message"].iloc[0])

    a = df[df.file == "a_ok.jpg"].iloc[0]
    b = df[df.file == "b_ok.jpg"].iloc[0]
    c = df[df.file == "c_nodir.jpg"].iloc[0]
    check("方向 90° 归类 front（bearing≈90）", a["view"] == "front")
    check("方向 0° 归类 left", b["view"] == "left")
    check("无方向填空位（right 为下一个空位）", c["view"] == "right")
    check("影像按契约命名保存", (out / f"{pid0}_front.jpg").exists()
          and (out / f"{pid0}_left.jpg").exists())

    # EXIF 读取单元验证
    gps = read_exif_gps(photos / "a_ok.jpg")
    check("EXIF 经纬度回读误差 < 1e-4",
          abs(gps["lat"] - (lat0 + 0.00001)) < 1e-4 and abs(gps["lon"] - lon0) < 1e-4)
    check("EXIF 方向回读", abs(gps["direction"] - 90) < 0.5)
    check("classify_view 占满返回 None",
          classify_view(90, 90, {"left", "right", "front", "back"}) is None)

    print("== 补录模板与地图 ==")
    check("生成 manual_coords.csv 模板", (out / "manual_coords.csv").exists())
    check("模板含 d_nogps.jpg", "d_nogps.jpg" in (out / "manual_coords.csv").read_text(encoding="utf-8-sig"))
    check("生成选点地图 HTML", (out / "ingest_pick_map.html").exists())

    print("== 第二轮：人工补录（坐标 + point_id 两种方式）==")
    # 新加两张无 GPS 照片：一张补坐标，一张补另一个点的 point_id
    pid1 = points.iloc[1]["point_id"]
    make_photo(photos / "i_manual_xy.jpg")
    make_photo(photos / "j_manual_pid.jpg")
    import pandas as pd
    pd.DataFrame([
        {"filename": "d_nogps.jpg", "lat": lat0, "lon": lon0, "point_id": ""},
        {"filename": "i_manual_xy.jpg", "lat": lat0, "lon": lon0, "point_id": ""},
        {"filename": "j_manual_pid.jpg", "lat": "", "lon": "", "point_id": pid1},
    ]).to_csv(tmp / "manual.csv", index=False, encoding="utf-8-sig")

    df2 = ingest_photos(gpkg, photos, out, snap_max_m=50, manual_csv=tmp / "manual.csv")
    st2 = dict(zip(df2["file"], df2["status"]))
    check("补录坐标成功 ok_manual", st2["d_nogps.jpg"] == "ok_manual")
    check("补录 point_id 成功 ok_manual", st2["j_manual_pid.jpg"] == "ok_manual")
    # point_1 的 4 视角被 a/b/c 占 3 个，d 补第 4 个，i 应判 duplicate
    check("视角占满判 duplicate", st2["i_manual_xy.jpg"] == "duplicate")
    check("续导跳过已导入文件",
          st2["a_ok.jpg"] == "already_imported"
          and st2["b_ok.jpg"] == "already_imported"
          and st2["c_nodir.jpg"] == "already_imported")

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n结果：{len(PASS) - n_fail}/{len(PASS)} 通过，{n_fail} 失败")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
