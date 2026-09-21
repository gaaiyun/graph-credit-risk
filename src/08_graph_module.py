"""图谱风险分模块 GRS（Graph Risk Score）。
第一阶段：只用训练期，拟合 Logit( y ~ 自身控制变量 + 行业 + 15 个预设图谱变量 )，
          GRS = 图谱变量部分的线性预测值（对数几率尺度，已剔除与自身财务重叠的部分）。
第二阶段：(a) GBM(主模型特征 + GRS)；(b) 叠加：logit(主模型PD) + GRS，不再拟合任何参数。
口径：滚动外推 2019-2023（每折重新拟合两阶段）+ 固定切分（测试 2022-2023）。"""
import numpy as np
import pandas as pd
import importlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *

M = importlib.import_module("04_model")
P = M.P
G_REL = ["sup", "cus", "holder", "invest", "ctrl", "person", "affil", "rp_other", "codef", "debtor"]
for r in G_REL:
    P[f"g_{r}"] = np.log1p((P[f"nb_{r}_fin"] + P[f"nb_{r}_trade"]).clip(lower=0))
P["g_ppr3"] = np.log1p(P.ppr3.clip(lower=0))
P["g_hop2"] = np.log1p(P.hop2_all.clip(lower=0))
P["g_group"] = np.log1p(P.group_fin.clip(lower=0))
P["g_peer"] = P.peer_fin_rate
P["g_relgrowth"] = P.rel_rev_growth.clip(-2, 2)
GF = [f"g_{r}" for r in G_REL] + ["g_ppr3", "g_hop2", "g_group", "g_peer", "g_relgrowth"]
CTRL = ["lev", "roa", "cfo_to_debt", "cash_ratio", "log_ta", "impair_to_rev", "own_trade", "own_other_def", "own_viol",
        "firm_age", "soe", "fund_n", "top1_stake"]


def fit_grs(trn_mask, C=0.1):
    Xc = P[CTRL].astype(float)
    lo, hi = Xc[trn_mask].quantile(0.01), Xc[trn_mask].quantile(0.99)
    Xc = Xc.clip(lo, hi, axis=1)
    Xc = Xc.fillna(Xc[trn_mask].median())
    Xg = P[GF].astype(float).fillna(0)
    mu = pd.concat([Xc, Xg], axis=1)[trn_mask].mean()
    sd = pd.concat([Xc, Xg], axis=1)[trn_mask].std().replace(0, 1)
    Z = (pd.concat([Xc, Xg], axis=1) - mu) / sd
    Z = pd.concat([Z, pd.get_dummies(P.sw_l1, prefix="ind", dtype=float)], axis=1)
    lr = LogisticRegression(C=C, max_iter=5000)
    lr.fit(Z[trn_mask], P.y[trn_mask])
    beta = pd.Series(lr.coef_[0], index=Z.columns)
    grs = Z[GF].values @ beta[GF].values
    return grs, beta[GF]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run(trn, val, tst, tag):
    grs, beta = fit_grs(P.index.isin(trn.index))
    P["grs"] = grs
    trn, val = P.loc[trn.index], P.loc[val.index]
    out = P.loc[tst.index, ["code", "t", "y"]].copy()
    for base, cols in [("M1", M.SETS["M1 自身"]), ("S0", M.OWN_EV)]:
        p0, _, _ = M.fit_lgb(cols, data=(trn, val), seeds=M.SEEDS[:3])
        pg, _, _ = M.fit_lgb(cols + ["grs"], data=(trn, val), seeds=M.SEEDS[:3])
        out[f"{base}"] = p0[tst.index]
        out[f"{base}+GRS(特征)"] = pg[tst.index]
        out[f"{base}+GRS(叠加)"] = 1 / (1 + np.exp(-(logit(p0[tst.index]) + grs[P.index.get_indexer(tst.index)])))
    out["GRS单独"] = grs[P.index.get_indexer(tst.index)]
    print(tag, {c: round(roc_auc_score(out.y, out[c]), 4) for c in out.columns[3:]}, flush=True)
    return out, beta


def summarize(df, label):
    y = df.y.values
    rng = np.random.default_rng(3)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    cols = [c for c in df.columns if c not in ("code", "t", "y")]
    rows = []
    for c in cols:
        rows.append({"口径": label, "模型": c, "AUC": roc_auc_score(y, df[c]), "KS": M.ks(y, df[c].values),
                     "AP": M.average_precision_score(y, df[c]), "Top10%捕获率": M.capture(y, df[c].values)})
    res = pd.DataFrame(rows)
    pairs = [("M1+GRS(特征)", "M1"), ("M1+GRS(叠加)", "M1"), ("S0+GRS(特征)", "S0"), ("S0+GRS(叠加)", "S0")]
    d = {p: [] for p in pairs}
    for b in range(2000):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for a_, b_ in pairs:
            d[(a_, b_)].append(roc_auc_score(y[idx], df[a_].values[idx]) - roc_auc_score(y[idx], df[b_].values[idx]))
    for a_, b_ in pairs:
        x = np.array(d[(a_, b_)])
        m = res.模型 == a_
        res.loc[m, "ΔAUC"] = x.mean()
        res.loc[m, "ΔAUC_lo"] = np.percentile(x, 2.5)
        res.loc[m, "ΔAUC_hi"] = np.percentile(x, 97.5)
        res.loc[m, "P(Δ≤0)"] = (x <= 0).mean()
    return res


if __name__ == "__main__":
    roll, betas = [], []
    for k in [2019, 2020, 2021, 2022, 2023]:
        o, b = run(P[P.t <= k - 2], P[P.t == k - 1], P[P.t == k], f"fold {k}")
        roll.append(o); betas.append(b.rename(k))
    roll = pd.concat(roll, ignore_index=True)
    fixed, beta_fixed = run(P[P.t.isin(TRAIN_YEARS)], P[P.t.isin(VALID_YEARS)], P[P.t.isin(TEST_YEARS)], "fixed")
    res = pd.concat([summarize(roll, "滚动外推2019-2023"), summarize(fixed, "固定切分2022-2023")], ignore_index=True)
    res.to_csv(OUT / "grs_results.csv", index=False, encoding="utf-8-sig")
    print(res.round(4).to_string(index=False))
    bt = pd.concat(betas + [beta_fixed.rename("fixed")], axis=1)
    bt.to_csv(OUT / "grs_beta.csv", encoding="utf-8-sig")
    print(bt.round(3).to_string())
    roll.to_parquet(WORK / "grs_rolling.parquet", index=False)
    fixed.to_parquet(WORK / "grs_fixed.parquet", index=False)
    P[["code", "t", "grs"]].to_parquet(WORK / "grs_fixed_all.parquet", index=False)
