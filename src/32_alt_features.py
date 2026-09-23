"""补充数据的清洗与对齐：事件表、主体所在省份、非上市主体的工商登记属性。

输入（均为本项目自行取得）：
- cninfo_events.parquet（28）：巨潮公告标题风险事件；剔除“展期”这类大多与违约无关的标题，“被动减持”剔除“被动稀释”；
- H 盘 CSMAR 特殊处理变动表：进入 ST / *ST 的公告日；
- pledge_panel.parquet（29）：月末股权质押比例；
- 工商注册数据过滤结果（31a、31b）：登记状态、省市区县、注册资本、实缴资本、成立与核准日期、参保人数、国标行业门类；
- H 盘地级市与区县界线的属性表：由企业名称开头的地名推断所在省份（工商数据匹配不上时使用）。
产出 cache/work/alt_events.parquet、node_loc.parquet、node_registry.parquet。
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from common import *

SHORT = re.compile(r"(省|市|壮族自治区|回族自治区|维吾尔自治区|自治区|特别行政区)$")
short_prov = lambda x: SHORT.sub("", str(x).strip())

# ---------------------------------------------------------------- 1. 事件表
C = pd.read_parquet(WORK / "cninfo_events.parquet")
C = C[~C.resolve]
C = C[~((C.kind == "bond_default") & (C.kw == "展期"))]                           # 展期多为质押回购、员工持股计划等常规事项
C = C[~((C.kind == "pledge_liq") & C.title.str.contains("稀释") & ~C.title.str.contains("平仓|拍卖|强制|违约|司法"))]
# 主体是控股股东或实际控制人、且没提到公司本身的标题，单独归为“控股股东出险”（公告由上市公司发布，风险在股东层面）
SH = re.compile(r"控股股东|实际控制人|实控人|大股东|以上股东|一致行动人")
CO = re.compile(r"公司及|公司、|本公司|及公司|公司被|公司银行|公司部分银行|公司主要银行|子公司")
to_sh = C.kind.isin(["shixin", "bankruptcy", "debt_overdue", "bond_default"]) & C.title.str.contains(SH) & ~C.title.str.contains(CO)
C.loc[to_sh, "kind"] = "sh_distress"
ev = [C[["code", "date", "kind"]]]

ST_SRC = Path(r"H:\A股特殊处理\A股_特殊处理变动文件\A股_特殊处理变动文件\SPT_Trdchg.csv")
ST_CACHE = EXT / "csmar_st_new.parquet"                                         # H 盘是移动硬盘，离线时读 G 盘缓存
if ST_SRC.exists():
    st = pd.read_csv(ST_SRC, dtype=str)
    st = st[st.Chgtype.str[-1].isin(["B", "D"]) & ~st.Chgtype.str[0].isin(["B", "D"])]  # 从正常进入 ST / *ST
    st = pd.DataFrame({"code": st.Stkcd.str.zfill(6), "date": pd.to_datetime(st.Annoudt, errors="coerce")})
    st.to_parquet(ST_CACHE, index=False)
else:
    st = pd.read_parquet(ST_CACHE)
    print("[ST] H 盘未连接，使用缓存", ST_CACHE.name)
ev.append(st.assign(kind="st_new"))
EV = pd.concat(ev, ignore_index=True).dropna(subset=["date"]).drop_duplicates()
EV.to_parquet(WORK / "alt_events.parquet", index=False)
print("[事件]", EV.groupby("kind").agg(条数=("code", "size"), 公司=("code", "nunique")).to_string(), sep="\n")

# ---------------------------------------------------------------- 2. 工商登记属性（31a 生成名单、31b 号 R 脚本过滤的结果）
reg_files = sorted((EXT / "registry").glob("Gongshang_*.csv"))
REG = pd.concat([pd.read_csv(f, dtype=str) for f in reg_files], ignore_index=True) if reg_files else pd.DataFrame()
if len(REG):
    REG["key"] = REG.key.map(norm_name)
    REG["prov"] = REG["所属省份"].map(short_prov)
    REG["cap"] = pd.to_numeric(REG["注册资本"].str.extract(r"([\d.]+)")[0], errors="coerce")
    REG["cap_paid"] = pd.to_numeric(REG["实缴资本"].str.extract(r"([\d.]+)")[0], errors="coerce")
    REG["est"] = pd.to_datetime(REG["成立日期"], errors="coerce")
    REG["approve"] = pd.to_datetime(REG["核准日期"], errors="coerce")
    REG["insured"] = pd.to_numeric(REG["参保人数"], errors="coerce")
    REG["dead"] = REG["登记状态"].fillna("").str.contains("注销|吊销|撤销|清算|停业")
    REG = REG.sort_values(["key", "dead"]).drop_duplicates("key")               # 同名多条时优先保留存续记录
    print(f"[工商] {len(reg_files)} 个分片，匹配 {len(REG):,} 家；注销/吊销等 {int(REG.dead.sum()):,}")

# ---------------------------------------------------------------- 3. 所在省份
CITY_SHP, COUNTY_SHP = Path(r"H:\数据-Map_China\2023市级\2023年初地级市矢量cp936.shp"), Path(r"H:\区划\区划\县.shp")
PLACE_CACHE = EXT / "place_prov.json"
if CITY_SHP.exists() and COUNTY_SHP.exists():
    city = pyogrio.read_dataframe(CITY_SHP, encoding="cp936", read_geometry=False)
    county = pyogrio.read_dataframe(COUNTY_SHP, read_geometry=False)
    place = {}
    for nm, pv in zip(county.NAME, county["省"]):                               # 区县名（重名的不用）
        place.setdefault(str(nm), set()).add(short_prov(pv))
    for nm, pv in zip(city["地名"], city["省级"]):
        place.setdefault(str(nm), set()).add(short_prov(pv))
        base = re.sub(r"(市|地区|盟|自治州|藏族羌族自治州|.族自治州)$", "", str(nm))
        if len(base) >= 2:
            place.setdefault(base, set()).add(short_prov(pv))
    for pv in city["省级"].unique():
        place.setdefault(str(pv), set()).add(short_prov(pv))
        place.setdefault(short_prov(pv), set()).add(short_prov(pv))
    place = {k: next(iter(v)) for k, v in place.items() if len(v) == 1 and len(k) >= 2}
    json.dump(place, open(PLACE_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
elif PLACE_CACHE.exists():
    place = json.load(open(PLACE_CACHE, encoding="utf-8"))
    print("[定位] H 盘未连接，使用地名缓存", PLACE_CACHE.name)
else:
    print("[定位] H 盘未连接且无地名缓存：保留已有的 node_loc.parquet 与 node_registry.parquet，只更新事件表")
    raise SystemExit(0)
keys = sorted(place, key=len, reverse=True)
pat = re.compile("|".join(map(re.escape, keys)))


def guess_prov(name):
    m = re.search(r"[(（]([^()（）]{2,8})[)）]", name)                           # 括号里的地名，如“艾坦姆合金(山东)有限公司”
    if m and m.group(1) in place:
        return place[m.group(1)]
    m = pat.match(name)                                                        # 名称开头的地名
    return place[m.group(0)] if m else None


E = pd.read_parquet(WORK / "edges.parquet", columns=["src", "dst"])
nodes = pd.unique(pd.concat([E.src, E.dst]))
master = pd.read_parquet(WORK / "firm_master.parquet")
listed_prov = dict(zip("C:" + master.code, master.province.map(lambda x: short_prov(x) if isinstance(x, str) else None)))
rows = []
reg_prov = dict(zip("E:" + REG.key, REG.prov)) if len(REG) else {}
for n in nodes:
    if n.startswith("C:"):
        rows.append((n, listed_prov.get(n), "上市公司注册地"))
    else:
        p = reg_prov.get(n)
        if p:
            rows.append((n, p, "工商登记"))
        else:
            g = guess_prov(n[2:])
            rows.append((n, g, "名称推断" if g else None))
LOC = pd.DataFrame(rows, columns=["node", "prov", "src"])
LOC.to_parquet(WORK / "node_loc.parquet", index=False)
print("[定位]", LOC.src.value_counts(dropna=False).to_dict(), f"；有省份 {LOC.prov.notna().mean():.1%}")

if len(REG):
    R = REG.assign(node="E:" + REG.key)[["node", "prov", "所属城市", "所属区县", "dead", "approve", "est", "cap", "cap_paid", "insured", "企业规模", "国标行业门类", "登记状态", "统一社会信用代码"]]
    R = R[R.node.isin(set(nodes))]
    R.to_parquet(WORK / "node_registry.parquet", index=False)
    print(f"[工商→图谱] {len(R):,} 个非上市主体有登记信息，其中注销/吊销等 {int(R.dead.sum()):,}")
