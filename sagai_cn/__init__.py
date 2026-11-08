"""SAGAI-CN: Streetscape Analysis with Generative AI — 国内化版本.

流水线：路网采样 -> 百度全景下载 -> VLM 打分 -> 空间聚合制图。
上游项目: https://github.com/perezjoan/SAGAI (Apache 2.0)
"""

__version__ = "1.0.0"

# 数据契约（与上游 SAGAI 保持一致，改造/扩展时不得破坏）：
# - GeoPackage 固定两层: "streets"(street_id/highway/name/length_m) 与
#   "points"(point_id/street_id/Latitude/Longitude/bearing)
# - 影像命名: {point_id}_{left|right|front|back}.jpg，空图加 _NA 后缀
# - 评分 CSV: 每行一张影像，image_name 列 + 每任务一列
# - 坐标纪律: 成果库一律 WGS84/CGCS2000；GCJ-02/BD-09 仅用于发请求
