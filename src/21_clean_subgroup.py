"""预设子样本：决策时点前 36 个月自身没有任何被诉（金融/经营/其他）、违规记录的"干净"企业。
这类企业的自身信号只剩财务报表，最接近信息稀薄的经营类客户。无论结果如何都报告。
静态：滚动外推 2019-2023 的年度预测；动态：月度预测（剔除近 12 个月自身有欠款被诉/执行/违规的企业-月）。"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
clean_cols = ["own_trade", "own_other_def", "own_viol", "own_fin_old", "own_sec_fraud"]
P["clean"] = (P[clean_cols].fillna(0).sum(1) == 0)
rng = np.random.default_rng(5)


def boot(df, y, a, b, B=1000):
    firms = df.code.unique()
    g = df.groupby("code").indices
    yy, pa, pb = df[y].values, df[a].values, df[b].values
    out = []
    for _ in range(B):
        idx = np.concatenate([g[f] for f in rng.choice(firms, len(firms))])
        if 0 < yy[idx].sum() < len(idx):
            out.append(roc_auc_score(yy[idx], pa[idx]) - roc_auc_score(yy[idx], pb[idx]))
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


rows = []
S = pd.read_parquet(WORK / "oot_rolling.parquet").merge(P[["code", "t", "clean"]], on=["code", "t"])
for grp, d in [("干净企业", S[S.clean]), ("有自身负面记录", S[~S.clean])]:
    for a, b in [("M2 +一跳关联", "M1 自身"), ("M4 +结构位置", "M1 自身")]:
        m, lo, hi = boot(d, "y", a, b)
        rows.append({"口径": "年度(滚动外推)", "子样本": grp, "样本": len(d), "违约": int(d.y.sum()), "比较": f"{a} vs {b}",
                     "基线AUC": roc_auc_score(d.y, d[b]), "新模型AUC": roc_auc_score(d.y, d[a]), "ΔAUC": m, "lo": lo, "hi": hi})
R = pd.read_parquet(WORK / "dyn_oot.parquet")
DY = pd.read_parquet(WORK / "dyn_panel.parquet", columns=["code", "t", "k", "own_trade_12m", "own_enf_12m", "own_viol_12m"])
R = R.merge(DY, on=["code", "t", "k"]).merge(P[["code", "t", "clean"]], on=["code", "t"])
R["clean_m"] = R.clean & (R[["own_trade_12m", "own_enf_12m", "own_viol_12m"]].fillna(0).sum(1) == 0)
names = [c for c in R.columns if c[:2] in ("D1", "D2", "D3", "D4") and not c.startswith("D1T")]
d1, d2, d3, d4 = sorted(names)
d1t = [c for c in R.columns if c.startswith("D1T")][0]
for grp, d in [("干净企业", R[R.clean_m]), ("有自身负面记录", R[~R.clean_m])]:
    for a, b in [(d4, d1), (d3, d1), (d1t, d1)]:
        m, lo, hi = boot(d, "y6", a, b, B=500)
        rows.append({"口径": "月度动态(滚动外推)", "子样本": grp, "样本": len(d), "违约": int(d.y6.sum()), "比较": f"{a} vs {b}",
                     "基线AUC": roc_auc_score(d.y6, d[b]), "新模型AUC": roc_auc_score(d.y6, d[a]), "ΔAUC": m, "lo": lo, "hi": hi})
res = pd.DataFrame(rows)
res.to_csv(OUT / "clean_subgroup.csv", index=False, encoding="utf-8-sig")
print(res.round(4).to_string(index=False))
