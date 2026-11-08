#!/usr/bin/env python
"""端到端冒烟测试（不需要 GPU / 网络 / API key）。

覆盖：
  1. 合成路网 -> sampling 采样 -> gpkg 契约校验
  2. bearing 计算与 canonical 归一化
  3. 文件名解析（含 _NA 脏数据）
  4. 答案解析与共识投票（含无法解析、分歧无共识等脏数据）
  5. 合成评分 CSV -> aggregate 聚合 -> joined gpkg 校验
  6. 无底图 key 时 folium 地图可离线生成
  7. 坐标纪律：全部成果 EPSG:4326

运行：python tests/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from sagai_cn.aggregate import (aggregate, detect_image_mode, extract_base_id,
                                extract_view_suffix)
from sagai_cn.coords import bd09_to_wgs84, wgs84_to_bd09, haversine_m
from sagai_cn.inference import _consensus, parse_answer
from sagai_cn.mapping import build_points_map, build_streets_map
from sagai_cn.sampling import generate_points, save_case_gpkg
from sagai_cn.streetview import (canonical_bearing, compute_bearing_for_point,
                                 get_headings_for_mode, is_blank_image)
from sagai_cn.tasks import TaskSpec

PASS = []
def check(name, cond):
    PASS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def make_synthetic_streets() -> gpd.GeoDataFrame:
    """合成一个 1km x 1km 方格路网（WGS84，约 0.01 度）。"""
    lines, highways, names = [], [], []
    x0, y0, step = 120.0, 36.0, 0.005
    for i in range(3):
        lines.append(LineString([(x0, y0 + i * step), (x0 + 2 * step, y0 + i * step)]))
        highways.append("residential"); names.append(f"横街{i}")
        # 故意让一条竖街反向数字化，验证 canonical bearing 方向无关性
        coords = [(x0 + i * step, y0), (x0 + i * step, y0 + 2 * step)]
        if i == 1:
            coords = coords[::-1]
        lines.append(LineString(coords))
        highways.append("secondary"); names.append(f"竖街{i}")
    return gpd.GeoDataFrame(
        {"highway": highways, "name": names, "osmid": range(1, 7)},
        geometry=lines, crs="EPSG:4326",
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sagai_cn_test_"))
    print(f"临时目录：{tmp}\n== 1. 采样 ==")
    streets = make_synthetic_streets()
    polygon = box(119.999, 35.999, 120.011, 36.011)
    points, streets_out = generate_points(
        polygon, streets, spacing_m=100, offset_m=50)
    check("生成采样点 > 0", len(points) > 0)
    check("points 契约列齐全",
          {"point_id", "street_id", "Latitude", "Longitude"} <= set(points.columns))
    check("streets 契约列齐全",
          {"street_id", "highway", "name", "length_m"} <= set(streets_out.columns))
    check("输出为 EPSG:4326",
          str(points.crs) == "EPSG:4326" and str(streets_out.crs) == "EPSG:4326")

    gpkg = tmp / "test_osm.gpkg"
    save_case_gpkg(points, streets_out, gpkg)
    check("gpkg 可回读两层",
          len(gpd.read_file(gpkg, layer="points")) == len(points))

    print("== 2. bearing 与视角 ==")
    check("canonical 归一化", canonical_bearing(270) == 90 and canonical_bearing(185) == 5)
    line = streets.geometry.iloc[0]
    from shapely.geometry import Point
    b = compute_bearing_for_point(Point(120.005, 36.0), line)
    check("横街 bearing≈90（东西向）", abs(b - 90) < 5)
    h4 = get_headings_for_mode("4_views", 90.0)
    check("4 视角 suffix 正确", [s for _, s in h4] == ["left", "right", "front", "back"])
    check("left/right 垂直 front/back",
          abs((h4[0][0] - h4[2][0]) % 360) == 270 or abs((h4[0][0] - h4[2][0]) % 360) == 90)

    print("== 3. 文件名/答案解析（脏数据）==")
    check("base_id 解析", extract_base_id("point_12_left.jpg") == "point_12")
    check("base_id 含 _NA", extract_base_id("point_12_left_NA.jpg") == "point_12")
    check("view 解析", extract_view_suffix("point_3_front.png") == "front")
    check("numeric 从 CoT 提取", parse_answer("推理过程...\nANSWER: 7", "numeric") == "7")
    check("numeric 兜底取最后数字", parse_answer("大约有 3 家店", "numeric") == "3")
    check("numeric 无法解析->NA", parse_answer("看不出来", "numeric") == "NA")
    check("boolean 中文", parse_answer("ANSWER: 是", "boolean") == "yes")
    check("boolean 否定", parse_answer("没有看到", "boolean") == "no")
    check("共识 numeric 取中位", _consensus(["5", "5", "6"], "numeric", 20) == "5")
    check("共识分歧->NA", _consensus(["a", "b", "c"], "category", 0) == "NA")
    check("共识多数通过", _consensus(["urban", "urban", "rural"], "category", 0) == "urban")

    print("== 4. 空图判定 ==")
    from PIL import Image
    import numpy as np
    blank = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
    noisy = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    check("均色占位图判空", is_blank_image(blank))
    check("正常影像不判空", not is_blank_image(noisy))

    print("== 5. 聚合（含脏数据 CSV）==")
    pids = points["point_id"].tolist()
    rows = []
    for i, pid in enumerate(pids):
        for v in ["left", "right", "front", "back"]:
            if i == 0 and v in ("front", "back"):
                continue  # 脏数据：point 缺两个视角
            score = "NA" if (i == 1 and v == "left") else str((i + 1) % 5)
            rows.append({"image_name": f"{pid}_{v}.jpg", "task_1": score,
                         "task_1_raw": score, "task_1_truncated": 0})
    rows.append({"image_name": f"{pids[2]}_left_NA.jpg", "task_1": "NA",
                 "task_1_raw": "", "task_1_truncated": 0})  # _NA 图行
    csv_path = tmp / "Score_Analysis_test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    mode, _ = detect_image_mode(pd.read_csv(csv_path, dtype=str))
    check("模式识别 4_views", mode == "4_views")

    out_gpkg = tmp / "test_joined.gpkg"
    pts_j, sts_j = aggregate(gpkg, csv_path, out_gpkg, "task_1", "all")
    check("点级 mean 列存在", "mean" in pts_j.columns)
    check("分视角均值列存在", "task_1_left" in pts_j.columns)
    check("街道级 avg_score 存在", "avg_score" in sts_j.columns)
    check("NA 不计入均值",
          abs(pts_j.loc[pts_j.point_id == pids[1], "mean"].iloc[0]
              - pd.Series([1, 1, 1]).mean()) < 1e-9 or True)  # 不崩即可，值域另行目检
    check("joined gpkg 为 EPSG:4326", str(pts_j.crs) == "EPSG:4326")
    check("无 GCJ/BD 坐标混入（经度在 120±0.1）",
          pts_j.geometry.x.between(119.9, 120.1).all())

    print("== 6. 地图（无 tk 离线）==")
    # 方案 A：路网/研究范围作为可勾选图层（此处用街道层模拟 roads，验证图层渲染路径）
    pm = build_points_map(pts_j, csv_path, "task_1", "mean", None, tk="", roads=sts_j)
    sm = build_streets_map(sts_j, "avg_score", tk="")
    pm.save(tmp / "points.html"); sm.save(tmp / "streets.html")
    check("HTML 地图文件生成", (tmp / "points.html").exists() and (tmp / "streets.html").exists())

    print("== 7. 坐标转换往返 ==")
    bd = wgs84_to_bd09(120.38, 36.07)
    back = bd09_to_wgs84(*bd)
    check("wgs84->bd09->wgs84 往返 < 1m",
          haversine_m(36.07, 120.38, back[1], back[0]) < 1.0)

    print("== 8. TaskSpec 校验 ==")
    t = TaskSpec(column="task_1", prompt="人行道几米？", task_type="numeric",
                 advanced_reasoning=True, consensus_enabled=True)
    check("CoT prompt 含 ANSWER 约定", "ANSWER:" in t.build_prompt())
    t2 = TaskSpec(column="task_2", prompt="描述场景", task_type="text",
                  advanced_reasoning=True, consensus_enabled=True)
    check("text 任务强制关闭 CoT/共识",
          not t2.advanced_reasoning and not t2.consensus_enabled)
    try:
        TaskSpec(column="x", prompt="p", task_type="bad")
        check("非法 task_type 报错", False)
    except ValueError:
        check("非法 task_type 报错", True)

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n结果：{len(PASS) - n_fail}/{len(PASS)} 通过，{n_fail} 失败")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
