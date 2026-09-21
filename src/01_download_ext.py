"""补充下载公开数据（akshare）：申万行业分类历史、上市公司全称、退市名单。结果缓存到 cache/ext。"""
import sys, time, traceback
from pathlib import Path
import pandas as pd
import akshare as ak
import requests
import urllib3

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "cache" / "ext"
OUT.mkdir(parents=True, exist_ok=True)

jobs = {
    "sh_main": lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
    "sh_kcb": lambda: ak.stock_info_sh_name_code(symbol="科创板"),
    "sz_a": lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
    "bj_a": lambda: ak.stock_info_bj_name_code(),
    "sh_delist": lambda: ak.stock_info_sh_delist(symbol="全部"),
    "sz_delist": lambda: ak.stock_info_sz_delist(symbol="终止上市公司"),
}
only = set(sys.argv[1:])
for name, fn in jobs.items():
    if only and name not in only:
        continue
    out = OUT / f"{name}.parquet"
    if out.exists():
        print(f"[skip] {name}"); continue
    t = time.time()
    try:
        df = fn()
        df = df.astype(str)
        df.to_parquet(out, index=False)
        print(f"[OK] {name} shape={df.shape} cols={list(df.columns)[:10]} {time.time()-t:.1f}s", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)

# 申万行业分类（含历次调整的计入日期）：akshare 的接口因对方站点证书链不完整会报 SSL 错误，这里直接取官方发布文件。
# 该站点证书链缺失中间证书，只能关闭校验；文件只作为 Excel 数据读取，不执行任何内容。
raw = {"sw_StockClassifyUse_stock.xls": ("https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls", False),
       "szse_list_raw.xlsx": ("https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1110&TABKEY=tab1", True)}
urllib3.disable_warnings()
for fname, (url, verify) in raw.items():
    out = OUT / fname
    if out.exists():
        print(f"[skip] {fname}"); continue
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, verify=verify)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"[OK] {fname} {len(r.content):,} bytes")
