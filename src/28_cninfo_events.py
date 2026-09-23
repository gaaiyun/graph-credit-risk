"""补充稀有数据：巨潮资讯网公告标题里的风险事件（2015—2025）。

按关键词全市场检索公告标题，逐年分页抓取，得到带日期的上市公司风险事件：
债务逾期、债券兑付违约、失信被执行、银行账户冻结、股份司法冻结、违规担保、被申请破产或重整、质押平仓、被立案调查。
“解除 / 移出 / 撤销 / 澄清 / 不存在”等化解或否认类标题单独标记，不计为风险发生。
巨潮是境内站点，直连（不走系统代理），每次请求间隔 0.6 秒。
产出 cache/work/cninfo_events.parquet：code、date、title、kind、resolve（是否化解类）。
"""
import json
import re
import time

import pandas as pd
import requests

from common import *

KW = {
    "debt_overdue": ["债务逾期", "贷款逾期", "借款逾期", "逾期债务", "债务违约"],
    "bond_default": ["未能按期兑付", "未能如期兑付", "兑付风险", "实质性违约", "构成违约", "债券违约", "延期兑付", "展期"],
    "shixin": ["失信被执行人"],
    "acct_frozen": ["账户被冻结", "银行账户冻结"],
    "share_frozen": ["司法冻结", "轮候冻结", "股份被冻结"],
    "illegal_guar": ["违规担保", "担保逾期", "逾期担保"],
    "bankruptcy": ["被申请破产", "破产重整", "预重整", "申请重整"],
    "pledge_liq": ["强制平仓", "被动减持", "质押违约", "平仓风险"],
    "csrc_probe": ["立案调查", "立案告知书"],
}
RESOLVE = re.compile(r"解除|移出|撤销|撤回|澄清|不存在|已偿还|已兑付|完毕|终止|驳回|化解|消除|结案|解冻")
YEARS_Q = range(2015, 2026)
OUTF = WORK / "cninfo_events.parquet"
RAWF = EXT / "cninfo_raw.jsonl"

s = requests.Session()
s.trust_env = False
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128 Safari/537.36",
                  "Origin": "https://www.cninfo.com.cn", "Referer": "https://www.cninfo.com.cn/new/fulltextSearch"})


def query(kw, y, page):
    d = {"pageNum": page, "pageSize": 30, "column": "szse", "tabName": "fulltext", "plate": "", "stock": "", "searchkey": kw,
         "secid": "", "category": "", "trade": "", "seDate": f"{y}-01-01~{y}-12-31", "sortName": "", "sortType": "", "isHLtitle": "true"}
    for attempt in range(4):
        try:
            return s.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=d, timeout=40).json()
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    return {}


done = set()
if RAWF.exists():                                                     # 断点续抓
    for ln in RAWF.open(encoding="utf-8"):
        r = json.loads(ln)
        done.add((r["kw"], r["y"]))
fout = RAWF.open("a", encoding="utf-8")
for kind, kws in KW.items():
    for kw in kws:
        for y in YEARS_Q:
            if (kw, y) in done:
                continue
            rows, page = [], 1
            while True:
                j = query(kw, y, page)
                ann = j.get("announcements") or []
                rows += [{"code": a.get("secCode"), "name": a.get("secName"), "title": re.sub(r"</?em>", "", a.get("announcementTitle") or ""),
                          "ts": a.get("announcementTime"), "id": a.get("announcementId")} for a in ann]
                time.sleep(0.6)
                if not j.get("hasMore") or not ann or page >= 120:
                    break
                page += 1
            fout.write(json.dumps({"kind": kind, "kw": kw, "y": y, "rows": rows}, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{kind:<13} {kw:<7} {y}  {len(rows):>5}", flush=True)
fout.close()

recs = []
for ln in RAWF.open(encoding="utf-8"):
    r = json.loads(ln)
    for a in r["rows"]:
        recs.append({**a, "kind": r["kind"], "kw": r["kw"]})
D = pd.DataFrame(recs).dropna(subset=["code", "ts"])
D["date"] = pd.to_datetime(D.ts, unit="ms").dt.normalize()
D["code"] = D.code.astype(str).str.zfill(6)
D["resolve"] = D.title.str.contains(RESOLVE)
D = D.drop_duplicates(["id", "kind"])[["code", "name", "date", "title", "kind", "kw", "resolve"]]
D.to_parquet(OUTF, index=False)
print(f"\n[cninfo] {len(D):,} 条（化解类 {int(D.resolve.sum()):,}），{D.code.nunique():,} 家公司")
print(D[~D.resolve].groupby("kind").agg(条数=("code", "size"), 公司=("code", "nunique")).to_string())
