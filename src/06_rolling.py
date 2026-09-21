"""滚动外推验证：对 k=2019..2023，训练 ≤k-2、验证 k-1（早停）、测试 k。汇总 5 个外推年份，
比较各模型的合并 AUC / 逐年平均 AUC，并对 M4-M1、S1-S0 做配对 bootstrap。"""
import numpy as np
import pandas as pd
import importlib
from sklearn.metrics import roc_auc_score
from common import *

M = importlib.import_module("04_model")
P = M.P
sets = {**M.SETS,
        "S0 无报表基线": M.OWN_EV,
        "S1 无报表+图谱": M.OWN_EV + M.HOP1 + M.MULTI + M.STRUCT,
        "R1 M4剔除诉讼衍生边": [c for c in M.SETS["M4 +结构位置"]
                          if not any(k in c for k in ["codef", "debtor", "creditor"])]}
oot = []
for k in [2019, 2020, 2021, 2022, 2023]:
    trn, val, tst = P[P.t <= k - 2], P[P.t == k - 1], P[P.t == k]
    r = tst[["code", "t", "y"]].copy()
    for name, cols in sets.items():
        p, _, _ = M.fit_lgb(cols, data=(trn, val), seeds=M.SEEDS[:3])
        r[name] = p[tst.index]
    oot.append(r)
    print(k, {n: round(roc_auc_score(r.y, r[n]), 4) for n in sets}, flush=True)
oot = pd.concat(oot, ignore_index=True)
oot.to_parquet(WORK / "oot_rolling.parquet", index=False)

y = oot.y.values
rows = []
for n in sets:
    per = oot.groupby("t").apply(lambda g: roc_auc_score(g.y, g[n]))
    rows.append({"模型": n, "合并AUC": roc_auc_score(y, oot[n]), "逐年平均AUC": per.mean(), "KS": M.ks(y, oot[n].values),
                 "AP": M.average_precision_score(y, oot[n]), "Top10%捕获率": M.capture(y, oot[n].values)})
res = pd.DataFrame(rows)
rng = np.random.default_rng(7)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
pairs = [(n, "M1 自身") for n in ["M2 +一跳关联", "M3 +多跳传导", "M4 +结构位置", "M5 +同业竞争", "R1 M4剔除诉讼衍生边"]] + \
        [("S1 无报表+图谱", "S0 无报表基线")]
d = {p: [] for p in pairs}
for b in range(2000):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for a_, b_ in pairs:
        d[(a_, b_)].append(roc_auc_score(y[idx], oot[a_].values[idx]) - roc_auc_score(y[idx], oot[b_].values[idx]))
for a_, b_ in pairs:
    x = np.array(d[(a_, b_)])
    m = res.模型 == a_
    res.loc[m, "对照"] = b_
    res.loc[m, "ΔAUC"] = x.mean()
    res.loc[m, "ΔAUC_lo"] = np.percentile(x, 2.5)
    res.loc[m, "ΔAUC_hi"] = np.percentile(x, 97.5)
    res.loc[m, "P(Δ≤0)"] = (x <= 0).mean()
res.to_csv(OUT / "rolling_oot.csv", index=False, encoding="utf-8-sig")
print(f"外推样本 {len(oot)}，正例 {int(y.sum())}")
print(res.round(4).to_string(index=False))
