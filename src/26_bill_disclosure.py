"""补充稀有数据：上海票据交易所「票据信息披露平台」的承兑人名单。

赛题要的“逾期”信息在公开渠道基本只剩这一处，且覆盖的正是非上市的经营类企业（带统一社会信用代码）：
- 信用信息未披露名单：平台仍在按月发布，2023-01 起共 40+ 期，每期约 3 千家（承兑人未按规定披露承兑与付款信息）；
- 承兑人逾期名单 / 持续逾期名单：票交所 2026-02 起不再单独发布，只能从 Web Archive 取回若干期存档。

产出 cache/work/bill_events.parquet：name（规范化全称）、uscc、date（名单截止或发布日）、kind（undisclosed / overdue）。
用法：python 26_bill_disclosure.py [--refresh]
"""
import argparse
import io
import re
import time
import urllib.parse
import zipfile

import fitz
import pandas as pd
import requests

from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--refresh", action="store_true", help="重新下载已存在的 PDF")
args = ap.parse_args()

RAW = EXT / "bill"
RAW.mkdir(parents=True, exist_ok=True)
API = "https://disclosure.cpisp.shcpe.com.cn/ent"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128 Safari/537.36",
      "Referer": "https://disclosure.cpisp.shcpe.com.cn/", "Content-Type": "application/json"}
WAYBACK = [                                                        # Web Archive 上仅存的几期逾期名单
    ("20220123160113", "https://disclosure.shcpe.com.cn/cpec-ent/file/20220105/1641374514920截至2021年12月31日持续逾期名单.pdf"),
    ("20221126190757", "https://disclosure.shcpe.com.cn/cpec-ent/file/20220209/1644399939383截至2022年1月31日持续逾期名单.pdf"),
    ("20220324015554", "https://disclosure.shcpe.com.cn//cpec-ent/file/20220303/1646294485937截至2022年2月28日持续逾期名单.pdf"),
    ("20230325225125", "https://disclosure.shcpe.com.cn/cpec-ent/file/20220906/1662450169317截至2022年8月31日持续逾期名单.pdf"),
    ("20240225122103", "https://disclosure.shcpe.com.cn/cpec-ent/file/20230804/1691140568154截至2023年7月31日承兑人逾期名单.pdf"),
]
USCC = re.compile(r"^[0-9A-Z]{18}$")
DATE_IN_TITLE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月")
CUTOFF = re.compile(r"截至\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
s = requests.Session()
s.verify = False
s.headers.update(UA)
requests.packages.urllib3.disable_warnings()


def fetch_list(type_code, size=100):
    j = s.post(f"{API}/public/article/list", json={"title": "", "current": 1, "size": size, "userType": "ADMIN", "typeCode": type_code}, timeout=40).json()
    return (j.get("data") or {}).get("dataList") or []


def download_attachment(article_id, attachment_id, path):
    r = s.post(f"{API}/public/article/detail/attachments/download", json={"articleId": article_id, "attachmentId": attachment_id}, timeout=120)
    r.raise_for_status()
    blob = r.content
    if blob[:2] == b"PK":                                          # 平台把 PDF 打成 zip
        z = zipfile.ZipFile(io.BytesIO(blob))
        blob = z.read(z.namelist()[0])
    path.write_bytes(blob)
    return len(blob)


def parse_names(pdf_path):
    """名单 PDF 是「序号 / 企业名称 / 统一社会信用代码」三行一组，按信用代码定位企业名。"""
    doc = fitz.open(pdf_path)
    rows, lines = [], []
    for page in doc:
        lines += [ln.strip() for ln in page.get_text().split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if USCC.match(ln):
            for j in range(i - 1, max(-1, i - 4), -1):             # 往回找最近的非数字行作为企业名
                nm = lines[j]
                if not nm.isdigit() and len(nm) >= 4 and "企业名称" not in nm and "信用代码" not in nm:
                    rows.append((nm, ln))
                    break
    return rows


recs = []
# ---------------------------------------------------------------- 1. 信用信息未披露名单（平台在售）
arts = fetch_list("delay-rel-settle-month-list")
print(f"[平台] 信用信息未披露名单 {len(arts)} 期：{arts[-1]['publishTime']} — {arts[0]['publishTime']}", flush=True)
for a in arts:
    m = DATE_IN_TITLE.search(a["title"])
    if not m:
        continue
    ym = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    pdf = RAW / f"undisclosed_{ym}.pdf"
    if args.refresh or not pdf.exists():
        det = s.post(f"{API}/public/article/detail", json={"articleId": a["articleId"]}, timeout=40).json()["data"]
        att = (det.get("attachments") or [None])[0]
        if not att:
            print("  无附件", a["title"])
            continue
        n = download_attachment(a["articleId"], att["attachmentId"], pdf)
        time.sleep(1.2)
    rows = parse_names(pdf)
    date = pd.Timestamp(a["publishTime"])
    recs += [{"name": nm, "uscc": u, "date": date, "ym": ym, "kind": "undisclosed"} for nm, u in rows]
    print(f"  {ym} {len(rows):>5} 家  {pdf.stat().st_size / 1e6:.1f} MB", flush=True)

# ---------------------------------------------------------------- 2. 承兑人逾期 / 持续逾期名单（Web Archive 存档）
for ts, url in WAYBACK:
    name = urllib.parse.unquote(url.split("/")[-1])
    pdf = RAW / ("overdue_" + re.sub(r"[^0-9一-龥]", "", name)[:24] + ".pdf")
    if args.refresh or not pdf.exists():
        r = requests.get(f"https://web.archive.org/web/{ts}id_/{url}", timeout=180, verify=False,
                         headers={"User-Agent": UA["User-Agent"]})
        if r.status_code != 200 or r.content[:4] != b"%PDF":
            print("  存档取回失败", r.status_code, name)
            continue
        pdf.write_bytes(r.content)
        time.sleep(2)
    m = CUTOFF.search(name)
    date = pd.Timestamp(f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}") if m else pd.NaT
    rows = parse_names(pdf)
    recs += [{"name": nm, "uscc": u, "date": date, "ym": date.strftime("%Y-%m"), "kind": "overdue"} for nm, u in rows]
    print(f"  {name[:28]} {len(rows):>5} 家", flush=True)

B = pd.DataFrame(recs)
B["name"] = B.name.map(norm_name)
B = B.drop_duplicates(["name", "uscc", "ym", "kind"])
B.to_parquet(WORK / "bill_events.parquet", index=False)
print(f"\n[bill] {len(B):,} 条，{B.name.nunique():,} 家企业；"
      f"未披露 {int((B.kind == 'undisclosed').sum()):,}（{B[B.kind == 'undisclosed'].ym.min()}—{B[B.kind == 'undisclosed'].ym.max()}），"
      f"逾期 {int((B.kind == 'overdue').sum()):,}（{B[B.kind == 'overdue'].ym.min()}—{B[B.kind == 'overdue'].ym.max()}）")
