"""为 31b 号 R 脚本生成工商注册数据的待匹配名单 cache/ext/registry/targets.txt。

名单 = 图谱里的非上市主体（规范化全称，长度 ≥ 5 且不是“某公司”“其他关联方”这类匿名名称）
     ∪ 票交所承兑人名单上的全部企业（带统一社会信用代码，名称规范化后并入）。
"""
import pandas as pd

from common import *

E = pd.read_parquet(WORK / "edges.parquet", columns=["src", "dst"])
nodes = pd.unique(pd.concat([E.src, E.dst]))
graph_names = {n[2:] for n in nodes if n.startswith("E:") and len(n) - 2 >= 5 and not is_anonymous(n[2:])}
bill_names = set(pd.read_parquet(WORK / "bill_events.parquet", columns=["name"]).name.map(norm_name))
targets = sorted(graph_names | bill_names)
out = EXT / "registry"
out.mkdir(parents=True, exist_ok=True)
(out / "targets.txt").write_text("\n".join(targets), encoding="utf-8")
print(f"[targets] 图谱非上市主体 {len(graph_names):,}，票据名单企业 {len(bill_names):,}，合计 {len(targets):,} 个名称")
