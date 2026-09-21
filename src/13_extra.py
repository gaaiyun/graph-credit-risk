"""两项补充分析（均基于滚动外推 2019-2023 的样本外预测，不重新训练）：
A. 与经营类客户最接近的子样本：民营（第一大股东非国资）且总资产低于当年中位数。
B. 贷中预警：自身模型评为中低风险（当年分位 < 80%）的企业中，股东/共同被告/债务人三条渠道出现风险邻居时的违约率。"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
oot = pd.read_parquet(WORK / "oot_rolling.parquet")
X = oot.merge(P[["code", "t", "soe", "log_ta", "nb_holder_risky", "nb_codef_risky", "nb_debtor_risky", "nb_cus_risky"]],
              on=["code", "t"], how="left")
med = P.groupby("t").log_ta.median()
X["small_private"] = (X.soe != 1) & (X.log_ta < X.t.map(med))
rng = np.random.default_rng(11)


def delta_ci(d, a, b, B=2000):
    y = d.y.values
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    xs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        xs.append(roc_auc_score(y[idx], d[a].values[idx]) - roc_auc_score(y[idx], d[b].values[idx]))
    xs = np.array(xs)
    return xs.mean(), np.percentile(xs, 2.5), np.percentile(xs, 97.5)


rows = []
for grp, d in [("全样本", X), ("民营且规模低于中位数", X[X.small_private]), ("其他", X[~X.small_private])]:
    for a, b in [("M2 +一跳关联", "M1 自身"), ("M4 +结构位置", "M1 自身"), ("S1 无报表+图谱", "S0 无报表基线")]:
        m, lo, hi = delta_ci(d, a, b)
        rows.append({"子样本": grp, "样本": len(d), "违约": int(d.y.sum()), "比较": f"{a} vs {b}",
                     "基线AUC": roc_auc_score(d.y, d[b]), "新模型AUC": roc_auc_score(d.y, d[a]), "ΔAUC": m, "ΔAUC_lo": lo, "ΔAUC_hi": hi})
sub = pd.DataFrame(rows)
sub.to_csv(OUT / "subgroup.csv", index=False, encoding="utf-8-sig")
print(sub.round(4).to_string(index=False))

# B. 预警触发
X["m1_pct"] = X.groupby("t")["M1 自身"].rank(pct=True)
X["trigger"] = (X[["nb_holder_risky", "nb_codef_risky", "nb_debtor_risky"]].fillna(0).sum(1) > 0)
low = X[X.m1_pct < 0.8]
tr = low.groupby("trigger").agg(企业年=("y", "size"), 违约=("y", "sum"), 违约率=("y", "mean"))
top = X[X.m1_pct >= 0.8]
res = {"中低风险段_触发_企业年": int(tr.loc[True, "企业年"]), "中低风险段_触发_违约": int(tr.loc[True, "违约"]),
       "中低风险段_触发_违约率": float(tr.loc[True, "违约率"]), "中低风险段_未触发_违约率": float(tr.loc[False, "违约率"]),
       "中低风险段_触发占比": float(low.trigger.mean()), "提升倍数": float(tr.loc[True, "违约率"] / tr.loc[False, "违约率"]),
       "中低风险段违约中被触发捕获比例": float(low[low.y == 1].trigger.mean()),
       "高风险段违约率": float(top.y.mean()), "中低风险段违约数": int(low.y.sum()), "全部违约数": int(X.y.sum())}
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(OUT / "trigger.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
