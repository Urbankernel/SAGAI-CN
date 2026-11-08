"""Block 6 国产化：天地图底图的 Folium 交互地图。

底图纪律：
  - 默认天地图（CGCS2000，有审图号，可直接叠加 WGS84 成果）
  - 不使用高德/百度瓦片（GCJ-02/BD-09 叠加 WGS84 点位会有数百米偏移）
  - 无天地图 tk 时退回无瓦片模式并明确提示，而不是悄悄用不合规底图
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from branca.element import Element, MacroElement
from jinja2 import Template

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from .aggregate import extract_base_id

TIANDITU_VEC = "https://t{s}.tianditu.gov.cn/DataServer?T=vec_w&x={x}&y={y}&l={z}&tk={tk}"
TIANDITU_CVA = "https://t{s}.tianditu.gov.cn/DataServer?T=cva_w&x={x}&y={y}&l={z}&tk={tk}"
TIANDITU_IMG = "https://t{s}.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk={tk}"
TIANDITU_CIA = "https://t{s}.tianditu.gov.cn/DataServer?T=cia_w&x={x}&y={y}&l={z}&tk={tk}"

# 路网 / 研究范围配色（图例与绘制共用，避免色值漂移）
_ROAD_CASING = "#212121"   # 路网底层描边（深色，保证与任意底图分离）
_ROAD_MAIN = "#e91e63"     # 路网顶层主线（品红，避开蓝点/橙面）
_AREA_EDGE = "#e65100"     # 研究范围描边（橙）
_AREA_FILL = "#ff9800"     # 研究范围填充（橙）


def score_to_hex(score, vmin: float, vmax: float, cmap_name: str) -> str:
    if pd.isna(score):
        return "#cccccc"
    cmap = plt.get_cmap(cmap_name)
    val = max(0.0, min(1.0, (float(score) - vmin) / max(vmax - vmin, 0.001)))
    r, g, b, _ = cmap(val)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _cmap_gradient_css(cmap_name: str, steps: int = 12) -> str:
    """把 matplotlib colormap 采样成 CSS linear-gradient（图例色带用）。"""
    cmap = plt.get_cmap(cmap_name)
    colors = []
    for i in range(steps):
        r, g, b, _ = cmap(i / (steps - 1))
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return "linear-gradient(to right, " + ", ".join(colors) + ")"


def _viridis_lut(cmap_name: str, steps: int = 32) -> list[str]:
    """把 colormap 采样成 hex 列表，供前端 JS 做颜色插值。"""
    cmap = plt.get_cmap(cmap_name)
    out = []
    for i in range(steps):
        r, g, b, _ = cmap(i / (steps - 1))
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


# 前端颜色插值工具（内联脚本，供切换着色列时复用）
_COLOR_SCRIPT = (
    "<script>\n"
    "function _sagaiHex2rgb(h){var n=parseInt(h.slice(1),16);"
    "return[(n>>16)&255,(n>>8)&255,n&255];}\n"
    "function _sagaiRgb2hex(r,g,b){return '#'+((1<<24)+(r<<16)+(g<<8)+b)"
    ".toString(16).slice(1);}\n"
    "var _sagaiLUT = %s;\n"
    "function _sagaiColor(t){"
    "if(t<=0)return _sagaiLUT[0];"
    "if(t>=1)return _sagaiLUT[_sagaiLUT.length-1];"
    "var x=t*(_sagaiLUT.length-1),i=Math.floor(x),f=x-i;"
    "var a=_sagaiHex2rgb(_sagaiLUT[i]),b=_sagaiHex2rgb(_sagaiLUT[i+1]);"
    "return _sagaiRgb2hex(Math.round(a[0]+(b[0]-a[0])*f),"
    "Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f));}\n"
    "</script>"
)


# 弹窗图片灯箱：点击弹窗内街景缩略图 → 全屏查看原图；点击任意处 / Esc 关闭。
# 用事件委托（document 级）而非逐图绑定——folium 弹窗内容是打开时才挂进 DOM，
# 且渲染脚本切换列时会重建弹窗，委托可天然覆盖动态创建的图片。
_IMAGE_VIEWER_SCRIPT = (
    "<script>\n(function(){\n"
    "function ensure(){var v=document.getElementById('sagai_viewer');"
    "if(v)return v;"
    "v=document.createElement('div');v.id='sagai_viewer';"
    "v.style.cssText='position:fixed;inset:0;z-index:99999;"
    "background:rgba(0,0,0,0.88);display:none;"
    "align-items:center;justify-content:center;cursor:zoom-out;';"
    "v.innerHTML='<img id=\"sagai_viewer_img\" alt=\"\" "
    "style=\"max-width:94vw;max-height:94vh;box-shadow:0 4px 30px rgba(0,0,0,0.6);"
    "border-radius:4px;\"/>';"
    "document.body.appendChild(v);"
    "v.addEventListener('click',function(){v.style.display='none';});"
    "document.addEventListener('keydown',function(e){"
    "if(e.key==='Escape')v.style.display='none';});"
    "return v;}\n"
    "document.addEventListener('click',function(e){var t=e.target;"
    "if(t&&t.classList&&t.classList.contains('sagai-popup-img')){"
    "var v=ensure();"
    "document.getElementById('sagai_viewer_img').src=t.src;"
    "v.style.display='flex';}});\n"
    "})();\n</script>"
)


def _legend_html(switch_cols: list[str], default_col: str,
                 cmap_name: str, gray_label: str | None = None,
                 layers_html: str = "") -> str:
    """图例卡片（含评分列切换下拉框 + 色带 + 0~vmax）。vmax 由前端切换时刷新。

    layers_html：可选的"图层"分组（路网/研究范围），由调用方在需要时拼好传入。
    """
    grad = _cmap_gradient_css(cmap_name)
    options = "".join(
        f'<option value="{c}"{" selected" if c == default_col else ""}>{c}</option>'
        for c in switch_cols
    )
    gray_html = ""
    if gray_label:
        gray_html = (
            '<div style="margin-top:8px;display:flex;align-items:center;gap:6px;">'
            '<span style="display:inline-block;width:12px;height:12px;'
            'background:#ffffff;border:2px solid #9e9e9e;border-radius:50%;"></span>'
            f"<span>{gray_label}</span></div>"
        )
    return (
        '<div id="sagai_legend" style="position: fixed; bottom: 28px; left: 12px;'
        'z-index: 9999; background: rgba(255,255,255,0.93); padding: 10px 12px;'
        'border-radius: 6px; font-family: Arial, sans-serif; font-size: 12px;'
        'color: #333; box-shadow: 0 1px 6px rgba(0,0,0,0.35);">'
        '<div style="font-weight: bold; margin-bottom: 4px;">评分列'
        '<select id="sagai_col_switch" style="margin-left:6px; font-size:12px;'
        'padding:1px 2px; border:1px solid #ccc; border-radius:3px;">'
        f"{options}</select></div>"
        f'<div style="width: 180px; height: 12px; background: {grad};'
        'border-radius: 2px;"></div>'
        '<div style="display: flex; justify-content: space-between;'
        'font-size: 11px; margin-top: 2px; color: #666;">'
        '<span>0</span><span id="sagai_v25"></span><span id="sagai_v50"></span>'
        '<span id="sagai_v75"></span><span id="sagai_vmax">0.00</span></div>'
        f"{gray_html}"
        f"{layers_html}</div>"
    )


def _points_script(map_var: str, pts_data: list, switch_cols: list[str],
                   default_col: str) -> str:
    """点级渲染/切换脚本：按底图类型（矢量/影像）动态套用两套样式。

    - 有分点：白描边实心圆（影像模式描边更粗 + 白色 halo 光晕衬底）
    - 无分点：空心圆（白底灰边，弱化但可辨）
    - hover 放大，tooltip 常显点号与当前列分数
    """
    data_json = json.dumps(pts_data, ensure_ascii=True)
    js = (
        "<script>\n(function(){\n"
        # 不在此处立即引用地图变量（见 build_points_map 注释），只存名字，
        # boot 时经 window[name] 解析。
        "var MAPNAME=@@MAPNAME@@;\n"
        "var DATA=@@DATA@@;\n"
        "var DEFAULT=@@DEFAULT@@;\n"
        "var MAP=null,markers=[],CURCOL=DEFAULT,IMG=false;\n"
        "function colorFor(v,vmax){if(v==null)return null;"
        "var t=vmax>0?v/vmax:0;return _sagaiColor(Math.min(1,Math.max(0,t)));}\n"
        "function fmt(v){return v==null?'N/A':v.toFixed(2);}\n"
        "function updateLegend(vmax){"
        "var m={'sagai_v25':0.25,'sagai_v50':0.5,'sagai_v75':0.75,'sagai_vmax':1.0};"
        "for(var k in m){var el=document.getElementById(k);"
        "if(el)el.textContent=(vmax*m[k]).toFixed(2);}}\n"
        "function mainStyle(d,col,vmax){var c=colorFor(d.cols[col],vmax);"
        "if(c!=null){return{radius:7,color:'#ffffff',weight:IMG?2.5:2,"
        "fillColor:c,fillOpacity:0.95};}"
        "return{radius:4,color:IMG?'#ffffff':'#9e9e9e',weight:2,"
        "fillColor:'#ffffff',fillOpacity:IMG?0.1:0.6};}\n"
        "function render(col){CURCOL=col;"
        "var vals=[],i;for(i=0;i<DATA.length;i++){"
        "var v=DATA[i].cols[col];if(v!=null)vals.push(v);}"
        "var vmax=vals.length?Math.max.apply(null,vals):1;"
        "for(i=0;i<DATA.length;i++){var d=DATA[i],st=mainStyle(d,col,vmax);"
        "var tip=d.pid+'<br>'+col+': '+fmt(d.cols[col]);"
        "if(!markers[i]){"
        "var halo=L.circleMarker([d.lat,d.lng],{radius:10,color:'transparent',"
        "weight:0,fillColor:'#ffffff',fillOpacity:0}).addTo(MAP);"
        "var main=L.circleMarker([d.lat,d.lng],st).bindPopup(d.popup,{maxWidth:450})"
        ".bindTooltip(tip,{sticky:true}).addTo(MAP);"
        "main._sagaiCur=st;"
        "main.on('mouseover',function(){this.setStyle({radius:this._sagaiCur.radius+2,"
        "weight:this._sagaiCur.weight+1});});"
        "main.on('mouseout',function(){this.setStyle(this._sagaiCur);});"
        "markers[i]={halo:halo,main:main};}"
        "else{markers[i].main._sagaiCur=st;markers[i].main.setStyle(st);"
        "markers[i].main.setTooltipContent(tip);"
        "markers[i].halo.setStyle({fillOpacity:IMG?0.25:0});}}"
        "updateLegend(vmax);}\n"
        "function boot(){MAP=window[MAPNAME];if(!MAP)return setTimeout(boot,60);"
        "MAP.on('baselayerchange',function(e){IMG=(e.name==='天地图影像');render(CURCOL);});"
        "var sel=document.getElementById('sagai_col_switch');"
        "if(sel)sel.addEventListener('change',function(){render(sel.value);});"
        "render(DEFAULT);}\n"
        "if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',boot);}"
        "else{boot();}\n"
        "})();\n</script>"
    )
    return (js.replace("@@MAPNAME@@", json.dumps(map_var))
               .replace("@@DATA@@", data_json)
               .replace("@@DEFAULT@@", json.dumps(default_col)))


def _streets_script(map_var: str, streets_data: list, switch_cols: list[str],
                    default_col: str) -> str:
    """街道级渲染/切换脚本：每条街画成 casing（白壳）+ core（色线）双层线。

    - 有分街道：白壳宽 8~9 + 色线宽 4~4.5（影像模式壳更宽）
    - 无分街道：白壳 + 灰色虚线，弱化可辨
    - hover 加粗，tooltip 常显街道与当前列分数
    """
    data_json = json.dumps(streets_data, ensure_ascii=True)
    js = (
        "<script>\n(function(){\n"
        "var MAPNAME=@@MAPNAME@@;\n"
        "var DATA=@@DATA@@;\n"
        "var DEFAULT=@@DEFAULT@@;\n"
        "var MAP=null,layers=[],CURCOL=DEFAULT,IMG=false;\n"
        "function colorFor(v,vmax){if(v==null)return null;"
        "var t=vmax>0?v/vmax:0;return _sagaiColor(Math.min(1,Math.max(0,t)));}\n"
        "function fmt(v){return v==null?'N/A':v.toFixed(2);}\n"
        "function updateLegend(vmax){"
        "var m={'sagai_v25':0.25,'sagai_v50':0.5,'sagai_v75':0.75,'sagai_vmax':1.0};"
        "for(var k in m){var el=document.getElementById(k);"
        "if(el)el.textContent=(vmax*m[k]).toFixed(2);}}\n"
        "function stylesFor(d,col,vmax){var c=colorFor(d.cols[col],vmax);"
        "if(c!=null){return{case:{color:'#ffffff',weight:IMG?9:8,opacity:IMG?0.85:0.9},"
        "core:{color:c,weight:IMG?4.5:4,opacity:0.9,dashArray:null}};}"
        "return{case:{color:'#ffffff',weight:IMG?9:8,opacity:0.8},"
        "core:{color:'#9e9e9e',weight:3,opacity:0.9,dashArray:'6,6'}};}\n"
        "function render(col){CURCOL=col;"
        "var vals=[],i;for(i=0;i<DATA.length;i++){"
        "var v=DATA[i].cols[col];if(v!=null)vals.push(v);}"
        "var vmax=vals.length?Math.max.apply(null,vals):1;"
        "for(i=0;i<DATA.length;i++){var d=DATA[i],s=stylesFor(d,col,vmax);"
        "var tip=(d.sid||'')+'<br>'+col+': '+fmt(d.cols[col]);"
        "if(!layers[i]){"
        "var caseL=L.geoJson(d.geom,{style:s.case}).addTo(MAP);"
        "var coreL=L.geoJson(d.geom,{style:s.core}).bindPopup(d.popup)"
        ".bindTooltip(tip,{sticky:true}).addTo(MAP);"
        "coreL._sagaiCur=s.core;"
        "coreL.on('mouseover',function(){this.setStyle({weight:this._sagaiCur.weight+2});});"
        "coreL.on('mouseout',function(){this.setStyle(this._sagaiCur);});"
        "layers[i]={caseL:caseL,coreL:coreL};}"
        "else{layers[i].coreL._sagaiCur=s.core;layers[i].caseL.setStyle(s.case);"
        "layers[i].coreL.setStyle(s.core);layers[i].coreL.setTooltipContent(tip);}}"
        "updateLegend(vmax);}\n"
        "function boot(){MAP=window[MAPNAME];if(!MAP)return setTimeout(boot,60);"
        "MAP.on('baselayerchange',function(e){IMG=(e.name==='天地图影像');render(CURCOL);});"
        "var sel=document.getElementById('sagai_col_switch');"
        "if(sel)sel.addEventListener('change',function(){render(sel.value);});"
        "render(DEFAULT);}\n"
        "if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',boot);}"
        "else{boot();}\n"
        "})();\n</script>"
    )
    return (js.replace("@@MAPNAME@@", json.dumps(map_var))
               .replace("@@DATA@@", data_json)
               .replace("@@DEFAULT@@", json.dumps(default_col)))


def _base_map(center_lat: float, center_lon: float, tk: str,
              imagery: bool = False, zoom: int = 14) -> folium.Map:
    """天地图底图：矢量/影像底图 + 各自注记层，可图层切换。

    注意：必须显式用 folium.TileLayer 并传 subdomains="01234567"（天地图子域
    是 t0~t7）。不要用 folium.Map(tiles=URL) —— 那条路径会把 subdomains 丢掉，
    回落到 folium 默认的 "abc"，使 {s} 被替换成 ta/tb/tc 等不存在的子域，
    导致底图瓦片 404、只剩注记能显示。

    LayerControl 不在这里添加 —— folium 要求 LayerControl 必须「最后」add，
    否则它的 JS 会先于 GeoJson/CircleMarker 渲染，引用到尚未赋值的变量，
    运行时抛错并中断后续所有图层渲染（表现为只有底图、没有成果要素和切换面板）。
    因此 LayerControl 由调用方（build_points_map / build_streets_map）在末尾 add。
    """
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    if not tk:
        print("[warn] 未配置 TIANDITU_TK，地图无底图瓦片（不会退回境外/偏移底图）")
        return m

    def _url(tpl: str) -> str:
        return tpl.format(s="{s}", x="{x}", y="{y}", z="{z}", tk=tk)

    common = dict(subdomains="01234567", max_zoom=18, max_native_zoom=18)

    # 两个底图（base layer，LayerControl 里单选切换）
    folium.TileLayer(_url(TIANDITU_VEC), name="天地图矢量", attr="天地图",
                     show=not imagery, **common).add_to(m)
    folium.TileLayer(_url(TIANDITU_IMG), name="天地图影像", attr="天地图",
                     show=imagery, **common).add_to(m)

    # 两个注记（overlay，可勾选叠加）
    folium.TileLayer(_url(TIANDITU_CVA), name="矢量注记", attr="天地图注记",
                     overlay=True, show=not imagery, **common).add_to(m)
    folium.TileLayer(_url(TIANDITU_CIA), name="影像注记", attr="天地图注记",
                     overlay=True, show=imagery, **common).add_to(m)

    return m


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """统一转 WGS84（无 CRS 或已是 4326 则原样返回）。

    每次生成地图时对路网/研究范围调用：用户提供的 roads.gpkg / study_area.gpkg
    可能是 CGCS2000 高斯投影（如 EPSG:4526）等非 4326 坐标系，不转换会与
    天地图底图（CGCS2000）和 WGS84 采样点错位数百米。
    """
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        return gdf.to_crs("EPSG:4326")
    return gdf


class _PaneScript(MacroElement):
    """在地图创建脚本之后、成果图层脚本之前创建自定义 pane。

    必须用 MacroElement（jinja script 宏）注入——直接挂 Element 不会被渲染，
    且注入点必须在带 pane 的图层之前，否则图层初始化时 pane 不存在会报错。
    """

    def __init__(self, map_name: str):
        super().__init__()
        self._template = Template(
            "{% macro script(this, kwargs) %}\n"
            f"{map_name}.createPane('sv-area').style.zIndex = 350;\n"
            f"{map_name}.createPane('sv-road').style.zIndex = 380;\n"
            "{% endmacro %}"
        )


def build_points_map(points: gpd.GeoDataFrame, scores_csv: Path | None,
                     score_col: str, pt_col: str,
                     images_dir: Path | None, tk: str,
                     cmap_name: str = "viridis",
                     roads: gpd.GeoDataFrame | None = None,
                     area: gpd.GeoDataFrame | None = None) -> folium.Map:
    # 可切换的着色列：points 表里实际存在的数值聚合列
    # （mean/sum/median 对应 --agg average/sum/median；var 为视角间分歧度）
    switch_cols = [c for c in ("mean", "sum", "median", "var") if c in points.columns]
    if pt_col in points.columns and pt_col not in switch_cols:
        switch_cols.append(pt_col)
    if not switch_cols:
        switch_cols = [pt_col]
    default_col = pt_col if pt_col in switch_cols else switch_cols[0]

    m = _base_map(points.geometry.y.mean(), points.geometry.x.mean(), tk)

    # 自定义 pane：固定图层叠放顺序。Leaflet 把所有 overlay 放进同一个 pane，
    # 同 pane 内按"加入顺序"叠放——取消再勾选路网会跑到采样点上面。给研究范围
    # /路网独立 pane 后 z-index 固定，与勾选顺序无关：底图(200) < 研究范围(350)
    # < 路网(380) < 采样点(默认 overlayPane 400)。
    m.add_child(_PaneScript(m.get_name()))

    # 研究范围（面）：最底层，透明填充 + 虚线描边，可勾选
    if area is not None and len(area):
        folium.GeoJson(
            _to_wgs84(area).to_json(),
            name="研究范围",
            pane="sv-area",
            style_function=lambda x: {
                "color": _AREA_EDGE, "weight": 2, "opacity": 0.9,
                "dashArray": "6,4",
                "fillColor": _AREA_FILL, "fillOpacity": 0.06},
        ).add_to(m)

    # 路网（线）：双描边 casing（深色描边 + 品红主线），包 FeatureGroup 保持图层控件单开关
    if roads is not None and len(roads):
        roads_wgs = _to_wgs84(roads)
        fg = folium.FeatureGroup(name="路网")
        # 底层描边：深色更宽，提供与浅色/深色底图的分离度
        folium.GeoJson(
            roads_wgs.to_json(),
            pane="sv-road",
            style_function=lambda x: {"color": _ROAD_CASING, "weight": 6, "opacity": 1.0},
        ).add_to(fg)
        # 顶层主线：品红高亮（避开蓝采样点、橙研究范围），悬停显示道路标签
        # 标签字段按优先级自动探测：路名 → 道路等级 → 兜底"道路 N"（保证悬停总有反馈）
        label_col = alias = None
        for cand, a in (("name", "路名"), ("name:zh", "路名"), ("NAME", "路名"),
                        ("road_name", "路名"), ("路名", "路名")):
            if cand in roads_wgs.columns:
                label_col, alias = cand, a
                break
        if label_col is None and "highway" in roads_wgs.columns:
            label_col, alias = "highway", "道路等级"
        if label_col is None:
            roads_wgs = roads_wgs.copy()
            roads_wgs["_rid"] = [f"道路 {i + 1}" for i in range(len(roads_wgs))]
            label_col, alias = "_rid", "道路"
        main_kwargs: dict = {}
        main_kwargs["tooltip"] = folium.GeoJsonTooltip(
            fields=[label_col], aliases=[alias], localize=True)
        folium.GeoJson(
            roads_wgs.to_json(),
            pane="sv-road",
            style_function=lambda x: {"color": _ROAD_MAIN, "weight": 3.5, "opacity": 1.0},
            **main_kwargs,
        ).add_to(fg)
        fg.add_to(m)

    scores_df = None
    if scores_csv and Path(scores_csv).exists():
        scores_df = pd.read_csv(scores_csv, dtype=str, encoding="utf-8-sig")
        scores_df["base_id"] = scores_df["image_name"].apply(extract_base_id)

    pts_data = []
    for _, row in points.iterrows():
        pid = row["point_id"]
        cols = {c: (None if pd.isna(row.get(c)) else float(row[c])) for c in switch_cols}

        img_html = ""
        if scores_df is not None and images_dir:
            parts = []
            for _, irow in scores_df[scores_df["base_id"] == pid].iterrows():
                fn = irow["image_name"]
                p = Path(images_dir) / fn
                if not p.exists():
                    p = Path(images_dir) / fn.replace("_NA", "")
                if p.exists():
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    # class="sagai-popup-img" 供灯箱脚本识别；src 本身是原图
                    # base64，缩略只是 CSS 限制尺寸，点击放大后即为原始分辨率
                    # width:100% + aspect-ratio:1 → 在 2 列网格里每张图等宽成正方形
                    tag = (f'<img src="data:image/jpeg;base64,{b64}" '
                           f'class="sagai-popup-img" title="点击放大" '
                           f'style="width:100%;aspect-ratio:1;object-fit:cover;'
                           f'border-radius:3px;cursor:zoom-in;"/>')
                else:
                    tag = '<small>[无图]</small>'
                s = irow.get(score_col, "N/A")
                parts.append(f'<div style="text-align:center;min-width:0;">'
                             f'{tag}<br><small style="word-break:break-all;">{fn}</small><br>'
                             f'<small>Score: <b>{s}</b></small></div>')
            # 固定 2 列网格：inline-block 流式排版会按弹窗宽度能塞几张塞几张
            # （4 张图会排成 3+1 不整齐），grid 强制 2×N
            img_html = ('<hr><div style="display:grid;'
                        'grid-template-columns:repeat(2,1fr);'
                        'gap:6px 10px;margin-top:6px;">'
                        + "".join(parts) + "</div>") if parts else ""

        score_rows = "".join(
            f'<tr><td>{c}</td><td>{"—" if v is None else f"{v:.3f}"}</td></tr>'
            for c, v in cols.items()
        )
        popup = (f'<div style="min-width:300px;font-family:Arial;">'
                 f'<h4 style="margin:0;">{pid}</h4>'
                 f'<b>Street:</b> {row.get("street_id", "")}<br>'
                 f'<table style="font-size:12px;border-collapse:collapse;margin:4px 0;">'
                 f'<tr><th style="text-align:left;">列</th>'
                 f'<th style="text-align:right;padding-left:8px;">分数</th></tr>'
                 f'{score_rows}</table>'
                 f'{img_html}</div>')

        pts_data.append({
            "pid": pid,
            "lat": float(row.geometry.y),
            "lng": float(row.geometry.x),
            "cols": cols,
            "popup": popup,
        })

    m.get_root().html.add_child(
        Element(_COLOR_SCRIPT % json.dumps(_viridis_lut(cmap_name))))
    # 弹窗街景图点击放大（灯箱），仅点级地图有图片弹窗
    m.get_root().html.add_child(Element(_IMAGE_VIEWER_SCRIPT))

    # 图例的"图层"分组：仅在有路网/研究范围图层时显示
    layers_html = ""
    if (roads is not None and len(roads)) or (area is not None and len(area)):
        layers_html = '<div style="font-weight: bold; margin: 8px 0 4px;">图层</div>'
    if roads is not None and len(roads):
        layers_html += (
            '<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="width:14px;height:3px;background:{_ROAD_MAIN};'
            f'display:inline-block;"></span>路网</div>')
    if area is not None and len(area):
        layers_html += (
            '<div style="display:flex;align-items:center;gap:6px;margin-top:3px;">'
            f'<span style="width:14px;height:10px;background:rgba(255,152,0,0.12);'
            f'border:1.5px dashed {_AREA_EDGE};display:inline-block;"></span>研究范围</div>')

    m.get_root().html.add_child(
        Element(_legend_html(switch_cols, default_col, cmap_name,
                             gray_label="无照片采样点", layers_html=layers_html)))
    m.get_root().html.add_child(
        Element(_points_script(m.get_name(), pts_data, switch_cols, default_col)))

    # LayerControl 必须最后 add（见 _base_map 注释），否则切换面板与成果图层会丢
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_streets_map(streets: gpd.GeoDataFrame, st_col: str, tk: str,
                      cmap_name: str = "viridis") -> folium.Map:
    # 可切换的着色列：streets 表里实际存在的数值聚合列
    # （avg/total/median_score 对应 --agg；var_score 为各点方差均值=分歧度预警）
    switch_cols = [c for c in ("avg_score", "total_score", "median_score",
                               "var_score") if c in streets.columns]
    if st_col in streets.columns and st_col not in switch_cols:
        switch_cols.append(st_col)
    if not switch_cols:
        switch_cols = [st_col]
    default_col = st_col if st_col in switch_cols else switch_cols[0]

    # 投影后算质心，避免地理坐标系下 centroid 不准（并有 UserWarning）
    cent = streets.to_crs(streets.estimate_utm_crs()).geometry.centroid.to_crs("EPSG:4326")
    m = _base_map(cent.y.mean(), cent.x.mean(), tk)

    streets_data = []
    for _, row in streets.iterrows():
        cols = {c: (None if pd.isna(row.get(c)) else float(row[c])) for c in switch_cols}
        score_rows = "".join(
            f'<tr><td>{c}</td><td>{"—" if v is None else f"{v:.3f}"}</td></tr>'
            for c, v in cols.items()
        )
        popup = (f'<b>{row.get("street_id", "")}</b><br>'
                 f'Name: {row.get("name", "") or ""}<br>'
                 f'Type: {row.get("highway", "") or ""}<br>'
                 f'<table style="font-size:12px;border-collapse:collapse;margin:4px 0;">'
                 f'<tr><th style="text-align:left;">列</th>'
                 f'<th style="text-align:right;padding-left:8px;">分数</th></tr>'
                 f'{score_rows}</table>')
        streets_data.append({
            "sid": row.get("street_id", ""),
            "geom": row.geometry.__geo_interface__,
            "cols": cols,
            "popup": popup,
        })

    m.get_root().html.add_child(
        Element(_COLOR_SCRIPT % json.dumps(_viridis_lut(cmap_name))))
    m.get_root().html.add_child(
        Element(_legend_html(switch_cols, default_col, cmap_name)))
    m.get_root().html.add_child(
        Element(_streets_script(m.get_name(), streets_data, switch_cols, default_col)))

    # LayerControl 必须最后 add（见 _base_map 注释）
    folium.LayerControl(collapsed=False).add_to(m)
    return m
