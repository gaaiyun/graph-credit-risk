"""逐层消融：M1 自身 → M2 +一跳 → M3 +多跳 → M4 +结构 → M5 +同业竞争。
LightGBM 为主模型（5 个随机种子平均），逻辑回归为对照；训练 2016-2020，验证 2021（早停），测试 2022-2023（时间外推）。
输出：各模型测试集指标 + 配对 bootstrap 置信区间、逐年 AUC、全样本预测、SHAP。"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
P["board_c"] = P.board.astype("category").cat.codes
P["sw_l1_c"] = P.sw_l1.astype("category").cat.codes

FIN = ["cur_ratio", "quick_ratio", "cash_ratio", "wc_to_loan", "int_debt_ratio", "lev", "eq_mult", "int_cover", "cfo_to_debt",
       "cfo_to_cl", "cfo_to_intdebt", "cfo_to_due", "tang_lev", "ltloan_to_ta", "roa", "roe", "gross_margin", "net_margin",
       "fin_exp_ratio", "impair_to_rev", "cfo_to_profit", "ebit_margin", "period_exp_ratio", "log_rev", "log_ta",
       "d_roa", "d_lev", "d_gross_margin", "d_cfo_to_debt", "d_cur_ratio", "rev_growth", "loss", "loss_2y"]
OWN_EV = ["own_fin_old", "own_trade", "own_other_def", "own_sec_fraud", "own_def_amt12", "own_def_n12", "own_viol",
          "own_viol_severe", "own_probe12", "sell_major", "sell_exec", "sell_person", "inst_ratio", "fund_n",
          "top1_stake", "top1_person", "soe", "top10_nonfin", "firm_age", "board_c", "sw_l1_c"]
REL = ["sup", "cus", "holder", "invest", "ctrl", "person", "affil", "rp_other", "codef", "debtor", "creditor"]
HOP1 = [f"nb_{r}_{k}" for r in REL for k in ["n", "fin", "trade", "risky", "dist"]] + ["hop1_all"]
MULTI = [f"hop2_{r}" for r in REL] + ["hop2_all", "ppr3"]
STRUCT = ["deg", "deg_listed", "pagerank", "kcore", "group_size", "group_listed", "group_risk", "group_fin"]
COMP = ["peer_n", "peer_fin_rate", "peer_loss_rate", "peer_roa", "peer_d_roa", "peer_rev_growth", "rel_roa", "rel_d_roa",
        "rel_rev_growth"]
SETS = {
    "M1 自身": FIN + OWN_EV,
    "M2 +一跳关联": FIN + OWN_EV + HOP1,
    "M3 +多跳传导": FIN + OWN_EV + HOP1 + MULTI,
    "M4 +结构位置": FIN + OWN_EV + HOP1 + MULTI + STRUCT,
    "M5 +同业竞争": FIN + OWN_EV + HOP1 + MULTI + STRUCT + COMP,
}
CAT = ["board_c", "sw_l1_c"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40, feature_fraction=0.6,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=8)
SEEDS = [11, 22, 33, 44, 55]

tr = P[P.t.isin(TRAIN_YEARS)]
va = P[P.t.isin(VALID_YEARS)]
te = P[P.t.isin(TEST_YEARS)]
print(f"train {len(tr)}/{tr.y.sum()}  valid {len(va)}/{va.y.sum()}  test {len(te)}/{te.y.sum()}")


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def capture(y, p, q=0.1):
    """风险最高的 q 比例样本中捕获的坏样本占全部坏样本比例"""
    k = int(np.ceil(len(p) * q))
    idx = np.argsort(-p)[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def metrics(y, p):
    return {"AUC": roc_auc_score(y, p), "KS": ks(y, p), "AP": average_precision_score(y, p),
            "Top10%捕获率": capture(y, p, 0.1)}


def fit_lgb(cols, label="y", seeds=SEEDS, data=None):
    trn, val = (tr, va) if data is None else data
    preds, models, iters = [], [], []
    for sd in seeds:
        dtr = lgb.Dataset(trn[cols], trn[label], categorical_feature=[c for c in CAT if c in cols], free_raw_data=False)
        dva = lgb.Dataset(val[cols], val[label], reference=dtr)
        m = lgb.train({**PARAMS, "seed": sd}, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        models.append(m); iters.append(m.best_iteration)
        preds.append(m.predict(P[cols], num_iteration=m.best_iteration))
    return np.mean(preds, axis=0), models, iters


def fit_lr(cols, label="y"):
    X = P[cols].astype(float)
    lo, hi = X[P.t.isin(TRAIN_YEARS)].quantile(0.01), X[P.t.isin(TRAIN_YEARS)].quantile(0.99)
    X = X.clip(lo, hi, axis=1)
    med = X[P.t.isin(TRAIN_YEARS)].median()
    miss = X.isna().astype(float).add_suffix("_na")
    miss = miss.loc[:, miss[P.t.isin(TRAIN_YEARS)].mean() > 0.01]
    X = X.fillna(med)
    mu, sd = X[P.t.isin(TRAIN_YEARS)].mean(), X[P.t.isin(TRAIN_YEARS)].std().replace(0, 1)
    X = pd.concat([(X - mu) / sd, miss], axis=1)
    best, bp = None, -1
    for C in [0.003, 0.01, 0.03, 0.1]:
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        lr.fit(X[P.t.isin(TRAIN_YEARS)], P.loc[P.t.isin(TRAIN_YEARS), label])
        a = roc_auc_score(P.loc[P.t.isin(VALID_YEARS), label], lr.predict_proba(X[P.t.isin(VALID_YEARS)])[:, 1])
        if a > bp:
            best, bp = lr, a
    return best.predict_proba(X)[:, 1]


if __name__ == "__main__":
    pred = P[["code", "t", "y", "y_big", "y_broad", "sw_l1", "event_date"]].copy()
    rows, iters_all, models_final = [], {}, None
    for name, cols in SETS.items():
        p, models, iters = fit_lgb(cols)
        pred[f"gbm::{name}"] = p
        iters_all[name] = iters
        pl = fit_lr(cols)
        pred[f"lr::{name}"] = pl
        mt = te.index
        r = {"模型": name, "特征数": len(cols), **{k: v for k, v in metrics(P.loc[mt, "y"].values, p[mt]).items()},
             "LR_AUC": roc_auc_score(P.loc[mt, "y"], pl[mt]),
             "验证AUC": roc_auc_score(P.loc[va.index, "y"], p[va.index])}
        rows.append(r)
        print(f"{name:12s} n={len(cols):3d} iters={iters} | test AUC={r['AUC']:.4f} KS={r['KS']:.4f} AP={r['AP']:.4f} "
              f"cap10={r['Top10%捕获率']:.3f} | LR AUC={r['LR_AUC']:.4f} | valid AUC={r['验证AUC']:.4f}", flush=True)
        models_final = models
    res = pd.DataFrame(rows)

    # 配对 bootstrap：测试集重抽样 2000 次，比较各模型相对 M1 的 AUC/KS 增量
    rng = np.random.default_rng(0)
    yt = P.loc[te.index, "y"].values
    B = 2000
    names = list(SETS)
    boots = {n: [] for n in names}
    ks_b = {n: [] for n in names}
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])   # 分层重抽样
        for n in names:
            pp = pred.loc[te.index, f"gbm::{n}"].values[idx]
            boots[n].append(roc_auc_score(yt[idx], pp))
            ks_b[n].append(ks(yt[idx], pp))
    base = np.array(boots[names[0]]); base_ks = np.array(ks_b[names[0]])
    ci = []
    for n in names:
        a = np.array(boots[n]); k_ = np.array(ks_b[n])
        d = a - base; dk = k_ - base_ks
        ci.append({"模型": n, "AUC_lo": np.percentile(a, 2.5), "AUC_hi": np.percentile(a, 97.5),
                   "ΔAUC": d.mean(), "ΔAUC_lo": np.percentile(d, 2.5), "ΔAUC_hi": np.percentile(d, 97.5),
                   "P(ΔAUC≤0)": float((d <= 0).mean()), "ΔKS": dk.mean(), "ΔKS_lo": np.percentile(dk, 2.5),
                   "ΔKS_hi": np.percentile(dk, 97.5)})
    res = res.merge(pd.DataFrame(ci), on="模型")
    res.to_csv(OUT / "ablation.csv", index=False, encoding="utf-8-sig")
    print(res.round(4).to_string(index=False))

    # 逐年 AUC（全部年份，训练年份为样本内）
    yr = []
    for t, g in pred.groupby("t"):
        row = {"t": t, "n": len(g), "pos": int(g.y.sum())}
        for n in names:
            row[n] = roc_auc_score(g.y, g[f"gbm::{n}"]) if g.y.nunique() > 1 else np.nan
        yr.append(row)
    yr = pd.DataFrame(yr)
    yr.to_csv(OUT / "auc_by_year.csv", index=False, encoding="utf-8-sig")
    print(yr.round(3).to_string(index=False))

    # 稳健性：替代标签（金额≥1000万；宽口径含经营性欠款）
    rob = []
    for lab in ["y_big", "y_broad"]:
        for n in [names[0], names[-1]]:
            p, _, _ = fit_lgb(SETS[n], label=lab, seeds=SEEDS[:3])
            rob.append({"标签": lab, "模型": n, "测试AUC": roc_auc_score(P.loc[te.index, lab], p[te.index]),
                        "测试KS": ks(P.loc[te.index, lab].values, p[te.index]), "测试正例": int(P.loc[te.index, lab].sum())})
            print(rob[-1], flush=True)
    pd.DataFrame(rob).to_csv(OUT / "robust_labels.csv", index=False, encoding="utf-8-sig")

    pred.to_parquet(WORK / "predictions.parquet", index=False)

    # SHAP（M5 全特征，第一个种子模型，测试集）
    full_cols = SETS[names[-1]]
    expl = shap.TreeExplainer(models_final[0])
    sv = expl.shap_values(P.loc[te.index, full_cols])
    sv = sv[1] if isinstance(sv, list) else sv
    shp = pd.DataFrame(sv, columns=full_cols, index=te.index)
    shp.to_parquet(WORK / "shap_test.parquet")
    imp = shp.abs().mean().sort_values(ascending=False)
    grp = {c: "自身" for c in FIN + OWN_EV}
    grp.update({c: "一跳关联" for c in HOP1}); grp.update({c: "多跳传导" for c in MULTI})
    grp.update({c: "结构位置" for c in STRUCT}); grp.update({c: "同业竞争" for c in COMP})
    gi = imp.groupby(imp.index.map(grp)).sum()
    print("SHAP 分组贡献占比:", (gi / gi.sum()).round(3).to_dict())
    print("SHAP top25:", imp.head(25).round(4).to_dict())
    imp.rename("mean_abs_shap").to_csv(OUT / "shap_importance.csv", encoding="utf-8-sig")
    json.dump({"iters": iters_all, "group_share": (gi / gi.sum()).to_dict()}, open(OUT / "model_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for i, m in enumerate(models_final):
        m.save_model(str(WORK / f"lgb_full_seed{i}.txt"))
