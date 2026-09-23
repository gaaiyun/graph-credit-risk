"""由股权质押明细按时点还原月末质押比例（2015-12—2025-12），并与东财月末快照交叉验证。

某公司某月末的质押比例 = 公告日 ≤ 月末、且实际解押日为空或晚于月末的各笔质押「占总股本比例」之和；
控股股东质押比例只加总 IS_CONTROL_SHAREHOLDER 为是的记录。
明细只含重要股东（持股 5% 以上）的质押，会略低于全口径快照，所以先在重叠期验证相关性再使用。
验证结果（output/pledge_check.json）：与快照的相关系数只有 0.36，快照≥30% 的企业-月里明细还原也≥30% 的只有 18%，
明细漏掉大量质押记录，这个还原方案不采用；分析用的质押面板由 29d 号脚本直接抓取的月末快照拼成。
本脚本只产出 cache/work/pledge_panel_detail.parquet 与验证记录，不改动 pledge_panel.parquet。
"""
import json

import numpy as np
import pandas as pd

from common import *

D = pd.read_parquet(EXT / "pledge_detail.parquet")
D["code"] = D.SECURITY_CODE.astype(str).str.zfill(6)
D["notice"] = pd.to_datetime(D.NOTICE_DATE, errors="coerce")
D["unfreeze"] = pd.to_datetime(D.ACTUAL_UNFREEZE_DATE, errors="coerce")
D["tsr"] = pd.to_numeric(D.PF_TSR, errors="coerce")
D["ctrl"] = D.IS_CONTROL_SHAREHOLDER.astype(str).isin(["1", "是", "True", "true"])
D = D.dropna(subset=["notice", "tsr"])
D = D[(D.tsr > 0) & (D.tsr <= 100)]
print(f"[明细] {len(D):,} 笔，{D.code.nunique():,} 家；控股股东 {int(D.ctrl.sum()):,} 笔；"
      f"已解押 {int(D.unfreeze.notna().sum()):,}；公告日 {D.notice.min():%Y-%m}—{D.notice.max():%Y-%m}")

months = pd.period_range("2015-12", "2025-12", freq="M")
me = months.to_timestamp(how="end").normalize()
mi = {p: i for i, p in enumerate(months)}


def month_idx(ts):                                                   # 第一个 ≥ 该日期的月末
    p = ts.dt.to_period("M")
    return p.map(lambda x: mi.get(x, np.nan if x > months[-1] else 0) if pd.notna(x) else np.nan)


D["i_on"] = month_idx(D.notice)
D["i_off"] = month_idx(D.unfreeze)
D.loc[D.unfreeze.notna() & (D.unfreeze.dt.normalize() > me[D.i_off.fillna(0).astype(int)].values), "i_off"] += 1   # 解押日晚于该月末 → 下月末起才失效
codes = sorted(D.code.unique())
ci = {c: j for j, c in enumerate(codes)}


def build(sub):
    M = np.zeros((len(codes), len(months) + 1))
    on = sub.dropna(subset=["i_on"])
    np.add.at(M, (on.code.map(ci).values, on.i_on.astype(int).values), on.tsr.values)
    off = on.dropna(subset=["i_off"])
    off = off[off.i_off <= len(months)]
    np.add.at(M, (off.code.map(ci).values, off.i_off.astype(int).values), -off.tsr.values)
    return np.clip(np.cumsum(M, axis=1)[:, :len(months)], 0, 100)


allp, ctrlp = build(D), build(D[D.ctrl])
P = pd.DataFrame({"code": np.repeat(codes, len(months)), "date": np.tile(me, len(codes)),
                  "pledge_detail": allp.ravel(), "pledge_ctrl": ctrlp.ravel()})
P = P[(P.pledge_detail > 0) | (P.pledge_ctrl > 0)]

snap = pd.read_parquet(WORK / "pledge_panel.parquet") if (WORK / "pledge_panel.parquet").exists() else pd.DataFrame()
check = {}
if len(snap):
    snap = snap[snap.columns.intersection(["code", "date", "pledge_ratio", "pledge_shares", "pledge_n"])]
    snap["ym"] = pd.to_datetime(snap.date).dt.to_period("M")
    P["ym"] = P.date.dt.to_period("M")
    j = snap.merge(P[["code", "ym", "pledge_detail"]], on=["code", "ym"], how="left").fillna({"pledge_detail": 0})
    check = {"重叠企业-月": int(len(j)), "相关系数": float(j[["pledge_ratio", "pledge_detail"]].corr().iloc[0, 1]),
             "快照中位数": float(j.pledge_ratio.median()), "明细还原中位数": float(j.pledge_detail.median()),
             "快照≥30%的月份里明细也≥30%的比例": float((j.loc[j.pledge_ratio >= 30, "pledge_detail"] >= 30).mean())}
    print("[验证]", {k: round(v, 3) if isinstance(v, float) else v for k, v in check.items()})
    last = snap.ym.max()
    later = P[P.ym > last].rename(columns={"pledge_detail": "pledge_ratio"})[["code", "date", "pledge_ratio"]]
    out = pd.concat([snap[["code", "date", "pledge_ratio"]], later], ignore_index=True)
    out = out.merge(P[["code", "date", "pledge_ctrl"]], on=["code", "date"], how="left")
    print(f"[合并] 快照 {len(snap):,} 行（至 {last}）+ 明细还原 {len(later):,} 行（{later.date.min():%Y-%m}—{later.date.max():%Y-%m}）")
else:
    out = P.rename(columns={"pledge_detail": "pledge_ratio"})[["code", "date", "pledge_ratio", "pledge_ctrl"]]
P.drop(columns=["ym"], errors="ignore").to_parquet(WORK / "pledge_panel_detail.parquet", index=False)
json.dump(check, open(OUT / "pledge_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[明细还原+快照] {len(out):,} 行，{out.code.nunique():,} 家（未采用，见文件头说明）")
