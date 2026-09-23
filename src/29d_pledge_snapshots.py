"""补抓股权质押月末快照（东财 RPT_CSDC_LIST，源自中国结算周度数据），直连、逐页、慢速。

29 号脚本经 akshare 抓到 2022-05 后被限流；这里直接请求同一报表，每月依次尝试月末前的几个周五，
页间隔 2.5 秒，结果与 29 号的快照同一格式、同一口径，落在 cache/ext/pledge/YYYYMM.parquet。
"""
import time

import pandas as pd
import requests

from common import *

RAW = EXT / "pledge"
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
s = requests.Session()
s.trust_env = False
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128 Safari/537.36",
                  "Referer": "https://data.eastmoney.com/gpzy/pledgeRatio.aspx"})


def get(d, page):
    p = {"sortColumns": "PLEDGE_RATIO", "sortTypes": "-1", "pageSize": "500", "pageNumber": page, "reportName": "RPT_CSDC_LIST",
         "columns": "ALL", "source": "WEB", "client": "WEB", "filter": f"(TRADE_DATE='{d:%Y-%m-%d}')"}
    for attempt in range(6):
        try:
            j = s.get(URL, params=p, timeout=30).json()
            return j.get("result")
        except Exception:
            time.sleep(min(120, 6 * 2 ** attempt))
    return None


for per in pd.period_range("2015-12", "2025-12", freq="M"):
    f = RAW / f"{per.strftime('%Y%m')}.parquet"
    if f.exists():
        continue
    end = per.to_timestamp(how="end").normalize()
    days = pd.date_range(end - pd.Timedelta(days=27), end)
    cands = sorted([d for d in days if d.weekday() == 4], reverse=True) + sorted([d for d in days if d.weekday() < 4], reverse=True)
    got = None
    for d in cands:
        r = get(d, 1)
        time.sleep(2.5)
        if r and r.get("count"):
            data = list(r["data"])
            for pg in range(2, (r.get("pages") or 1) + 1):
                rr = get(d, pg)
                data += (rr or {}).get("data") or []
                time.sleep(2.5)
            got = pd.DataFrame(data).assign(snap=d)
            break
    if got is None:
        print(f"{per} 无数据", flush=True)
        continue
    got = got.rename(columns={"SECURITY_CODE": "code", "PLEDGE_RATIO": "pledge_ratio", "REPURCHASE_BALANCE": "pledge_shares", "PLEDGE_DEAL_NUM": "pledge_n"})
    got["code"] = got.code.astype(str).str.zfill(6)
    got[["code", "snap", "pledge_ratio", "pledge_shares", "pledge_n"]].to_parquet(f, index=False)
    print(f"{per}  {len(got):>5} 家  快照 {got.snap.iloc[0]:%Y-%m-%d}", flush=True)

P = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("*.parquet"))], ignore_index=True).rename(columns={"snap": "date"})
P.to_parquet(WORK / "pledge_panel.parquet", index=False)
print(f"\n[pledge] {len(P):,} 行，{P.code.nunique():,} 家，{P.date.min():%Y-%m}—{P.date.max():%Y-%m}，{P.date.dt.to_period('M').nunique()} 个月")
