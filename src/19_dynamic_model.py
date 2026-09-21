"""月度动态图谱评分模型：逐层消融 + 滚动外推（测试 2019-2023 观察年的全部 12 个月）。
D1 自身（年度特征 + 自身事件时序 + 最新季报）→ D2 +静态图谱 → D3 +一跳关联事件时序 → D4 +多层级时序（兄弟企业/二跳/同业）；
D1T = D1 + 一跳与多层级时序（不含静态图谱），用于判断增量是否来自时序。
训练 ≤k-2 观察年；验证 k-1 年前 6 个月（其 6 个月标签窗不越过 D_k）；测试 k 年 12 个月。负样本按 30% 抽样训练。
置信区间：按企业整簇重抽样的 bootstrap（1000 次）。"""
import json
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import importlib
from common import *

M = importlib.import_module("04_model")
DY = pd.read_parquet(WORK / "dyn_panel.parquet")
P = M.P
STATIC_OWN = M.FIN + M.OWN_EV
STATIC_GRAPH = M.HOP1 + M.MULTI + M.STRUCT
_st = P[["code", "t"] + STATIC_OWN + STATIC_GRAPH].copy()
for _c in STATIC_OWN + STATIC_GRAPH:
    if _c not in M.CAT:
        _st[_c] = _st[_c].astype("float32")
DY = DY.merge(_st, on=["code", "t"], how="left")
del _st
for c in ["lev", "cur_ratio", "quick_ratio", "cash_ratio", "int_debt_ratio"]:
    q = {"lev": "q_lev", "cur_ratio": "q_cur", "quick_ratio": "q_quick", "cash_ratio": "q_cash", "int_debt_ratio": "q_intdebt"}[c]
    DY[f"dq_{c}"] = (DY[q] - DY[c]).astype("float32")
OWN_DYN = [c for c in DY.columns if re.fullmatch(r"own_(trade|enf|viol)_(1m|3m|12m|dec)", c)] + ["own_sell_6m", "k"] + \
          [c for c in DY.columns if c.startswith(("q_", "dq_"))]
REL = ["sup", "cus", "holder", "invest", "ctrl", "person", "affil", "rp_other", "codef", "debtor"]
DYN1 = [c for c in DY.columns if c.startswith(tuple(f"dy_{r}_" for r in REL)) or c.startswith("dy_all_")]
DYNM = [c for c in DY.columns if c.startswith(("dy_sib_", "dy_sc2hop", "dy_peer_"))] + ["sib_ctrl_n", "sib_pe_n"]
SETS = {"D1 自身(含自身时序)": STATIC_OWN + OWN_DYN,
        "D2 +静态图谱": STATIC_OWN + OWN_DYN + STATIC_GRAPH,
        "D3 +一跳关联事件时序": STATIC_OWN + OWN_DYN + STATIC_GRAPH + DYN1,
        "D4 +多层级时序": STATIC_OWN + OWN_DYN + STATIC_GRAPH + DYN1 + DYNM,
        "D1T 自身+关联时序(无静态图谱)": STATIC_OWN + OWN_DYN + DYN1 + DYNM}
PARAMS = dict(M.PARAMS, min_child_samples=80)
SEEDS = [11, 22]
LABEL = "y6"


def fit_fold(cols, tr_idx, va_idx, te_idx):
    """按行号只取所需列，转 float32 矩阵训练，避免整表复制"""
    cat = [i for i, c in enumerate(cols) if c in M.CAT]
    Xtr, Xva, Xte = (DY.loc[ix, cols].to_numpy(np.float32) for ix in (tr_idx, va_idx, te_idx))
    ytr, yva = DY.loc[tr_idx, LABEL].values, DY.loc[va_idx, LABEL].values
    preds, models = [], []
    for sd in SEEDS:
        dtr = lgb.Dataset(Xtr, ytr, feature_name=cols, categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train({**PARAMS, "seed": sd}, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        preds.append(m.predict(Xte, num_iteration=m.best_iteration))
        models.append(m)
    del Xtr, Xva, Xte
    return np.mean(preds, axis=0), models


def cluster_boot(df, a, b, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    firms = df.code.unique()
    g = df.groupby("code").indices
    ya, pa, pb = df[LABEL].values, df[a].values, df[b].values
    out = []
    for _ in range(B):
        idx = np.concatenate([g[f] for f in rng.choice(firms, len(firms))])
        if ya[idx].sum() == 0:
            continue
        out.append(roc_auc_score(ya[idx], pa[idx]) - roc_auc_score(ya[idx], pb[idx]))
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), (out <= 0).mean()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    res_rows, allp, models_keep = [], [], {}
    for k in [2019, 2020, 2021, 2022, 2023]:
        keep = (DY[LABEL].values == 1) | (rng.random(len(DY)) < 0.3)
        tr_idx = DY.index[(DY.t.values <= k - 2) & keep]
        va_idx = DY.index[(DY.t.values == k - 1) & (DY.k.values <= 5)]
        te_idx = DY.index[DY.t.values == k]
        tst = DY.loc[te_idx, ["code", "t", "k", "S", "y6", "y12", "def_date"]].copy()
        for name, cols in SETS.items():
            p, ms = fit_fold(cols, tr_idx, va_idx, te_idx)
            tst[name] = p
            models_keep[(k, name)] = ms
        allp.append(tst[["code", "t", "k", "S", "y6", "y12", "def_date"] + list(SETS)])
        print(k, {n: round(roc_auc_score(tst[LABEL], tst[n]), 4) for n in SETS}, flush=True)
    R = pd.concat(allp, ignore_index=True)
    R.to_parquet(WORK / "dyn_oot.parquet", index=False)
    y = R[LABEL].values
    rows = []
    for n in SETS:
        per = R.groupby("t").apply(lambda g: roc_auc_score(g[LABEL], g[n]), include_groups=False)
        rows.append({"模型": n, "特征数": len(SETS[n]), "AUC": roc_auc_score(y, R[n]), "逐年平均AUC": per.mean(), "KS": M.ks(y, R[n].values),
                     "AP": average_precision_score(y, R[n])})
    res = pd.DataFrame(rows)
    names = list(SETS)
    for a, b in [(names[1], names[0]), (names[2], names[1]), (names[3], names[1]), (names[3], names[0]), (names[4], names[0])]:
        d, lo, hi, p0 = cluster_boot(R, a, b)
        res.loc[res.模型 == a, f"Δ vs {b.split(' ')[0]}"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"{a} vs {b}: ΔAUC {d:+.4f} [{lo:+.4f}, {hi:+.4f}] P(Δ≤0)={p0:.3f}", flush=True)
        json.dump({"a": a, "b": b, "d": d, "lo": lo, "hi": hi, "p0": p0}, open(OUT / f"dyn_delta_{names.index(a)}_{names.index(b)}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
    res.to_csv(OUT / "dyn_ablation.csv", index=False, encoding="utf-8-sig")
    print(res.round(4).to_string(index=False))

    # 预警视角：每月把分数前 5% 的企业列入预警，统计违约事件在违约前 6 个月内是否被预警、提前量
    ev = []
    for n in [names[0], names[3]]:
        R["flag"] = R.groupby("S")[n].rank(pct=True) >= 0.95
        d = R[R.def_date.notna()].copy()
        d = d[(d.def_date > d.S) & (d.def_date <= d.S + pd.DateOffset(months=6))]
        grp = d.groupby(["code", "def_date"])
        hit = grp.flag.any()
        first = d[d.flag].groupby(["code", "def_date"]).S.min()
        lead = ((first.index.get_level_values(1) - first.values).days / 30.4375)
        prec = R[R.flag][LABEL].mean()
        ev.append({"模型": n, "违约事件数": int(len(hit)), "违约前6个月内被预警比例": float(hit.mean()),
                   "首次预警提前月数中位数": float(np.median(lead)) if len(lead) else np.nan, "预警名单6个月违约率": float(prec),
                   "基准6个月违约率": float(R[LABEL].mean())})
    ev = pd.DataFrame(ev)
    ev.to_csv(OUT / "dyn_alert.csv", index=False, encoding="utf-8-sig")
    print(ev.round(4).to_string(index=False))
    for (k, n), ms in models_keep.items():
        if n == names[3]:
            ms[0].save_model(str(WORK / f"dyn_D4_fold{k}.txt"))
        if n == names[0]:
            ms[0].save_model(str(WORK / f"dyn_D1_fold{k}.txt"))
    json.dump({"sets": {n: len(c) for n, c in SETS.items()}, "dyn1": DYN1, "dynm": DYNM, "own_dyn": OWN_DYN},
              open(OUT / "dyn_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
