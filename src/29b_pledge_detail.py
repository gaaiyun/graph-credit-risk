"""补充数据：东方财富「重要股东股权质押明细」（每笔质押一条，含公告日、质押起止与解押日、占总股本比例）。

月末快照接口在 2022 年后被限流时，曾打算用明细按时点还原任意月份的质押比例（29c）：
某月某公司的质押比例 = 公告日 ≤ 月末、且尚未解押的各笔质押「占总股本比例」之和。验证未通过（明细漏记太多），分析改用 29d 的快照，明细留作备查。
逐页抓取（每页 500 条），单页失败退避重试，已抓的页落盘可续抓；东财是境内站点，直连。
产出 cache/ext/pledge_detail.parquet。
"""
import json
import os
import time

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import pandas as pd
import requests

from common import *

RAW = EXT / "pledge_detail"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
s = requests.Session()
s.trust_env = False
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128 Safari/537.36",
                  "Referer": "https://data.eastmoney.com/gpzy/pledgeDetail.aspx"})


def page(n):
    params = {"sortColumns": "NOTICE_DATE", "sortTypes": "-1", "pageSize": "500", "pageNumber": n,
              "reportName": "RPTA_APP_ACCUMDETAILS", "columns": "ALL", "quoteColumns": "", "source": "WEB", "client": "WEB"}
    for attempt in range(8):
        try:
            j = s.get(URL, params=params, timeout=30).json()
            if j.get("result"):
                return j["result"]
        except Exception:
            pass
        time.sleep(min(120, 6 * 2 ** attempt))
    return None


first = page(1)
pages = first["pages"]
(RAW / "p0001.json").write_text(json.dumps(first["data"], ensure_ascii=False), encoding="utf-8")
print(f"[质押明细] 共 {pages} 页、{first['count']:,} 条", flush=True)
for n in range(2, pages + 1):
    f = RAW / f"p{n:04d}.json"
    if f.exists():
        continue
    res = page(n)
    if res is None:
        print(f"  第 {n} 页多次失败，先跳过", flush=True)
        continue
    f.write_text(json.dumps(res["data"], ensure_ascii=False), encoding="utf-8")
    if n % 20 == 0:
        print(f"  {n}/{pages}", flush=True)
    time.sleep(2.5)

rows = []
for f in sorted(RAW.glob("p*.json")):
    rows += json.loads(f.read_text(encoding="utf-8"))
D = pd.DataFrame(rows)
D.to_parquet(EXT / "pledge_detail.parquet", index=False)
print(f"\n[pledge_detail] {len(D):,} 条（{len(list(RAW.glob('p*.json')))}/{pages} 页），列：{list(D.columns)[:18]}")
