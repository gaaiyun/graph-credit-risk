"""补充稀有数据：巨潮资讯网“债券”栏目的违约与展期公告（2015—2025），覆盖不上市的发债主体。

28 号脚本只检索股票栏目，拿到的是上市公司自己的公告。发债的集团母公司、城投和民营控股股东多数不上市，
它们的兑付违约、展期和破产重整公告挂在债券栏目下（以深交所上市的公司债、企业债为主），标题里带发行人全称。
这里按关键词逐年检索债券栏目，从标题中取出出事主体的全称，规范化后与图谱节点逐字匹配（不做子串匹配）。
作为重整投资人参与、子公司或股东出事、年报延期披露、股票质押式回购违约这类标题不计入。
产出 cache/work/bond_events.parquet：node、date、title、kw（每条公告里出现的每个可匹配主体一行）。
"""
import json
import re
import time

import pandas as pd
import requests

from common import *

KW = ["违约", "未能按期", "未能如期", "未按期", "无法按期", "兑付风险", "偿付风险", "展期", "延期兑付", "破产", "重整"]
RAWF = EXT / "cninfo_bond_raw.jsonl"
NAME = re.compile(r"[一-龥（）()A-Za-z0-9]{4,40}?(?:股份有限公司|有限责任公司|集团有限公司|有限公司|集团公司)")
AGENT = re.compile(r"证券|评级|资信|信用评估|中诚信|鹏元|东方金诚|大公国际|联合信用|律师|会计师|受托|银行股份|交易所")

s = requests.Session()
s.trust_env = False
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128 Safari/537.36",
                  "Origin": "https://www.cninfo.com.cn", "Referer": "https://www.cninfo.com.cn/new/fulltextSearch"})


def query(kw, y, page):
    d = {"pageNum": page, "pageSize": 30, "column": "bond", "tabName": "fulltext", "plate": "", "stock": "", "searchkey": kw,
         "secid": "", "category": "", "trade": "", "seDate": f"{y}-01-01~{y}-12-31", "sortName": "", "sortType": "", "isHLtitle": "true"}
    for attempt in range(4):
        try:
            return s.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=d, timeout=40).json()
        except Exception:
            time.sleep(3 * (attempt + 1))
    return {}


done = set()
if RAWF.exists():
    for ln in RAWF.open(encoding="utf-8"):
        r = json.loads(ln)
        done.add((r["kw"], r["y"]))
fout = RAWF.open("a", encoding="utf-8")
for kw in KW:
    for y in range(2015, 2026):
        if (kw, y) in done:
            continue
        rows, page = [], 1
        while True:
            j = query(kw, y, page)
            ann = j.get("announcements") or []
            rows += [{"sec": a.get("secName"), "org": a.get("orgId"), "title": re.sub(r"</?em>", "", a.get("announcementTitle") or ""),
                      "ts": a.get("announcementTime"), "id": a.get("announcementId")} for a in ann]
            time.sleep(0.6)
            if not j.get("hasMore") or not ann or page >= 60:
                break
            page += 1
        fout.write(json.dumps({"kw": kw, "y": y, "rows": rows}, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"{kw:<5} {y}  {len(rows):>4}", flush=True)
fout.close()

recs = []
for ln in RAWF.open(encoding="utf-8"):
    r = json.loads(ln)
    recs += [{**a, "kw": r["kw"]} for a in r["rows"]]
D = pd.DataFrame(recs).dropna(subset=["ts"]).drop_duplicates("id")
D["date"] = pd.to_datetime(D.ts, unit="ms").dt.normalize()
RESOLVE = re.compile(r"解除|撤销|撤回|澄清|不存在|已偿还|已兑付|兑付完毕|完成兑付|执行完毕|终止|驳回|化解|消除|结案|摘牌|恢复")
NOISE = re.compile(r"参与|投资人|投资事项|投资框架|子公司|关联方|股东|控股上市公司|无法按期披露|年度报告|质押式回购|说明|资产重整")  # 作为投资人参与重整、股东或子公司出事等
D = D[~D.title.str.contains(RESOLVE) & ~D.title.str.contains(NOISE)]
D = D[~((D.kw == "展期") & D.title.str.contains("质押|回购|融资融券"))]            # 股票质押式回购的展期不是发债主体的展期

E = pd.read_parquet(WORK / "edges.parquet", columns=["src", "dst"])
nodes = set(pd.concat([E.src, E.dst]).unique())
ND = pd.read_parquet(WORK / "name_dict.parquet")                                  # 上市公司全称（含曾用名）→ 代码
full2code = {norm_name(n): "C:" + c for n, c in zip(ND.name, ND.code)}
SPLIT = re.compile(r"关于|关注|暨|及其|及|、|，|,|“|”|\"|《|》")


def subjects(title):
    """“甲关于乙……”的主体是乙；“甲关于‘某债券’违约的公告”后半没有企业全称时，主体是发布人甲。"""
    parts = title.split("关于", 1)
    names = {n for frag in SPLIT.split(parts[-1]) for n in NAME.findall(frag)}
    if not names and len(parts) == 2:
        names = {n for frag in SPLIT.split(parts[0]) for n in NAME.findall(frag)}
    return names


out = []
for r in D.itertuples():
    for nm in subjects(r.title):
        if AGENT.search(nm):
            continue
        k = norm_name(nm)
        node = full2code.get(k) or ("E:" + k if "E:" + k in nodes else None)
        if node:
            out.append((node, r.date, r.title, r.kw))
B = pd.DataFrame(out, columns=["node", "date", "title", "kw"]).drop_duplicates(["node", "date", "title"])
B.to_parquet(WORK / "bond_events.parquet", index=False)
print(f"\n[债券栏目] 公告 {len(D):,} 条；可匹配到图谱节点的 {len(B):,} 条，{B.node.nunique():,} 个主体"
      f"（上市 {B[B.node.str.startswith('C:')].node.nunique()}，非上市 {B[B.node.str.startswith('E:')].node.nunique()}）")
