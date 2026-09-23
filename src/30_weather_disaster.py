"""另类数据：自然灾害与极端天气，按「省 × 月」给出冲击指标（2015—2025）。

- 地震：USGS 地震目录，震级 ≥ 5.0，震中落在省界内；
- 台风：IBTrACS 西北太平洋最佳路径，热带风暴及以上（风速 ≥ 34 节）的路径点落在省界内；
- 气温与降水：NOAA 月度气候摘要（GSOM），中国境内气象站，按省平均后与同省同月份的多年均值比较，
  z ≥ 2 记为极端高温或极端降水，z ≤ -2 记为寒潮；另加绝对阈值（极端最高温 ≥ 40℃、单日最大降水 ≥ 100 毫米）。
省界来自 H 盘「数据-Map_China/2023省级」（审图号 GS(2019)1822 系列）。
产出 cache/work/prov_shocks.parquet 与 cache/work/weather_stations.parquet。
"""
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from common import *

RAW = EXT / "weather"
(RAW / "gsom").mkdir(parents=True, exist_ok=True)
PROV_SHP = Path(r"H:\数据-Map_China\2023省级\2023年初省级矢量cp936.shp")
s = requests.Session()
s.headers.update({"User-Agent": "graph-risk-model research (python-requests)"})

SHORT = re.compile(r"(省|市|壮族自治区|回族自治区|维吾尔自治区|自治区|特别行政区)$")


def short_prov(x):
    x = str(x).strip()
    return SHORT.sub("", x)


def get(url, dest, timeout=180):
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    for attempt in range(4):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return r.content
            if r.status_code == 404:
                return b""
        except Exception:
            time.sleep(3 * (attempt + 1))
    return b""


# ---------------------------------------------------------------- 省界
prov = gpd.read_file(PROV_SHP, encoding="cp936")
name_col = next(c for c in prov.columns if prov[c].astype(str).str.contains("北京|广东").any())
prov = prov[[name_col, "geometry"]].rename(columns={name_col: "prov_full"}).to_crs(4326)
prov["prov"] = prov.prov_full.map(short_prov)
print(f"[省界] {len(prov)} 个，字段 {name_col}：{prov.prov.tolist()[:6]} …")


def to_prov(df, lat="lat", lon="lon"):
    g = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon], df[lat])], crs=4326)
    j = gpd.sjoin(g, prov[["prov", "geometry"]], how="inner", predicate="within")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


# ---------------------------------------------------------------- 地震
eq = pd.read_csv(io.BytesIO(get("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2015-01-01&endtime=2026-01-01"
                                "&minlatitude=17&maxlatitude=54&minlongitude=73&maxlongitude=136&minmagnitude=5", RAW / "usgs_m5.csv")))
eq = eq.rename(columns={"latitude": "lat", "longitude": "lon"})
eq["ym"] = pd.to_datetime(eq.time).dt.strftime("%Y-%m")
eq = to_prov(eq)
q = eq.groupby(["prov", "ym"]).agg(quake_n=("mag", "size"), quake_mag=("mag", "max")).reset_index()
print(f"[地震] 境内 M≥5：{len(eq)} 次，涉及 {q.prov.nunique()} 省、{len(q)} 个省-月")

# ---------------------------------------------------------------- 台风
tf = RAW / "ibtracs_wp.csv"
get("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv", tf, timeout=900)
cols = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND", "CMA_WIND"]
tc = pd.read_csv(tf, usecols=cols, skiprows=[1], low_memory=False)
tc = tc[pd.to_numeric(tc.SEASON, errors="coerce") >= 2015]
for c in ["LAT", "LON", "WMO_WIND", "USA_WIND", "CMA_WIND"]:
    tc[c] = pd.to_numeric(tc[c], errors="coerce")
tc["wind"] = tc[["WMO_WIND", "USA_WIND", "CMA_WIND"]].max(axis=1)        # CMA 风速单位为米/秒以外时以 WMO/USA 节为准
tc = tc[tc.wind >= 34].rename(columns={"LAT": "lat", "LON": "lon"})
tc["ym"] = pd.to_datetime(tc.ISO_TIME).dt.strftime("%Y-%m")
tc = to_prov(tc)
t = tc.groupby(["prov", "ym"]).agg(typhoon_n=("SID", "nunique"), typhoon_wind=("wind", "max")).reset_index()
print(f"[台风] 登陆或过境路径点 {len(tc)} 个，{tc.SID.nunique()} 个台风，{len(t)} 个省-月")

# ---------------------------------------------------------------- 气象站月度气候（GSOM）
st_txt = get("https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt", RAW / "ghcnd-stations.txt").decode("utf-8", "ignore")
stations = pd.DataFrame([{"id": ln[0:11], "lat": float(ln[12:20]), "lon": float(ln[21:30]), "name": ln[41:71].strip()}
                         for ln in st_txt.splitlines() if ln.startswith("CH")])
stations = to_prov(stations)
print(f"[气象站] 中国境内 {len(stations)} 个，覆盖 {stations.prov.nunique()} 省", flush=True)


def load_station(sid):
    b = get(f"https://www.ncei.noaa.gov/data/global-summary-of-the-month/access/{sid}.csv", RAW / "gsom" / f"{sid}.csv", timeout=120)
    if not b:
        return None
    d = pd.read_csv(io.BytesIO(b), low_memory=False)
    keep = [c for c in ["DATE", "TAVG", "TMAX", "TMIN", "PRCP", "EMXT", "EMNT", "EMXP"] if c in d.columns]
    d = d[keep]
    d = d[d.DATE >= "2000-01"]
    d["id"] = sid
    return d


with ThreadPoolExecutor(max_workers=6) as ex:
    parts = [p for p in ex.map(load_station, stations.id) if p is not None and len(p)]
G = pd.concat(parts, ignore_index=True).merge(stations[["id", "prov"]], on="id")
G = G.rename(columns={"DATE": "ym"})
agg = G.groupby(["prov", "ym"]).agg(tmax=("TMAX", "mean"), tmin=("TMIN", "mean"), tavg=("TAVG", "mean"), prcp=("PRCP", "mean"),
                                    emxt=("EMXT", "max"), emxp=("EMXP", "max"), n_st=("id", "nunique")).reset_index()
agg["mon"] = agg.ym.str[5:7]
for v in ["tmax", "tmin", "prcp"]:
    grp = agg.groupby(["prov", "mon"])[v]
    agg[v + "_z"] = (agg[v] - grp.transform("mean")) / grp.transform("std")
agg["heat"] = (agg.tmax_z >= 2) | (agg.emxt >= 40)
agg["rain"] = (agg.prcp_z >= 2) | (agg.emxp >= 100)
agg["cold"] = agg.tmin_z <= -2
print(f"[气候] {G.id.nunique()} 站、{len(agg):,} 个省-月；极端高温 {int(agg.heat.sum())}、极端降水 {int(agg.rain.sum())}、寒潮 {int(agg.cold.sum())}")

# ---------------------------------------------------------------- 合并
grid = pd.MultiIndex.from_product([prov.prov.unique(), pd.period_range("2015-01", "2025-12", freq="M").strftime("%Y-%m")],
                                  names=["prov", "ym"]).to_frame(index=False)
S = grid.merge(q, how="left").merge(t, how="left").merge(agg[["prov", "ym", "tmax_z", "prcp_z", "tmin_z", "emxt", "emxp", "heat", "rain", "cold"]], how="left")
for c in ["quake_n", "typhoon_n"]:
    S[c] = S[c].fillna(0).astype(int)
for c in ["heat", "rain", "cold"]:
    S[c] = S[c].fillna(False).astype(bool)
S["quake"] = S.quake_n > 0
S["typhoon"] = S.typhoon_n > 0
S["disaster"] = S.quake | S.typhoon | S.rain                       # 突发性灾害：地震、台风、极端降水
S.to_parquet(WORK / "prov_shocks.parquet", index=False)
stations.to_parquet(WORK / "weather_stations.parquet", index=False)
print(f"\n[prov_shocks] {len(S):,} 个省-月；地震 {int(S.quake.sum())}、台风 {int(S.typhoon.sum())}、极端降水 {int(S.rain.sum())}、"
      f"极端高温 {int(S.heat.sum())}、寒潮 {int(S.cold.sum())}；任一突发灾害 {int(S.disaster.sum())}")
