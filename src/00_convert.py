r"""把 G:\data\data 下的 xlsx 一次性转成 parquet 缓存，后续脚本只读缓存。原始数据不动。"""
import sys, time
from pathlib import Path
import pandas as pd

RAW = Path(r"G:\data\data")
CACHE = Path(__file__).resolve().parents[1] / "cache"
CACHE.mkdir(exist_ok=True)

files = sorted(RAW.rglob("*.xlsx"))
for f in files:
    out = CACHE / (f.stem + ".parquet")
    if out.exists():
        print(f"[skip] {f.name}")
        continue
    t = time.time()
    # 全部按字符串读，避免股票代码前导零丢失；数值列后续按需转换
    df = pd.read_excel(f, dtype=str, engine="openpyxl")
    df.to_parquet(out, index=False)
    print(f"[OK] {f.relative_to(RAW)} -> {out.name}  shape={df.shape}  {time.time()-t:.1f}s", flush=True)
