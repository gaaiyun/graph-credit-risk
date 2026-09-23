"""补充数据：东方财富股权质押比例（akshare），2015-12—2025-12 每月末一期快照。

质押比例 = 质押股数 / 总股本。控股股东高比例质押说明股东自身资金链紧张，是股东渠道的前瞻信号，
也可作为“关联方（股东）压力”的量化版本。东财是境内站点，直连（NO_PROXY）。
快照落在 cache/ext/pledge/YYYYMM.parquet，已抓的月份跳过。经 akshare 抓到 2022-05 后被东财限流，剩余月份由 29d 号脚本直连补齐，
最终的 cache/work/pledge_panel.parquet 以 29d 的拼接结果为准：code、date、pledge_ratio、pledge_n、pledge_shares（万股）。
"""
import os
import time

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import requests

_orig_request = requests.sessions.Session.request


def _request_with_timeout(self, method, url, **kw):                   # akshare 内部请求不带超时，连接挂住会永远等下去
    kw.setdefault("timeout", 30)
    return _orig_request(self, method, url, **kw)


requests.sessions.Session.request = _request_with_timeout

import akshare as ak
import pandas as pd

from common import *

RAW = EXT / "pledge"
RAW.mkdir(parents=True, exist_ok=True)
cal = pd.Series(pd.bdate_range("2015-12-01", "2025-12-31"))          # 工作日；遇节假日由下面的回退处理（不依赖新浪日历接口）
month_ends = cal.groupby(cal.dt.to_period("M")).max().sort_values()


def fetch(d):
    for attempt in range(5):                                        # 东财偶尔重置连接：指数退避重试
        try:
            return ak.stock_gpzy_pledge_ratio_em(date=d.strftime("%Y%m%d"))
        except Exception as e:
            time.sleep(min(90, 5 * 2 ** attempt))
    return None


for me in month_ends:
    f = RAW / f"{me:%Y%m}.parquet"
    if f.exists():
        continue
    got = None
    for back in range(0, 12):                                       # 月末没有快照就往前找工作日
        d = cal[cal <= me].iloc[-1 - back]
        df = fetch(d)
        if df is not None and len(df):
            got = df.assign(snap=d)
            break
    if got is None:
        print(f"{me:%Y-%m} 无数据", flush=True)
        continue
    got = got.rename(columns={"股票代码": "code", "质押比例": "pledge_ratio", "质押股数": "pledge_shares", "质押笔数": "pledge_n"})
    got["code"] = got.code.astype(str).str.zfill(6)
    got[["code", "snap", "pledge_ratio", "pledge_shares", "pledge_n"]].to_parquet(f, index=False)
    print(f"{me:%Y-%m}  {len(got):>5} 家  快照 {got.snap.iloc[0]:%Y-%m-%d}", flush=True)
    time.sleep(3)

P = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("*.parquet"))], ignore_index=True)
P = P.rename(columns={"snap": "date"})
P.to_parquet(WORK / "pledge_panel.parquet", index=False)
print(f"\n[pledge] {len(P):,} 行，{P.code.nunique():,} 家，{P.date.min():%Y-%m}—{P.date.max():%Y-%m}")
