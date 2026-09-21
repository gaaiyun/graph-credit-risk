"""R-GCN 与同架构去边 MLP 的滚动外推（k=2019..2023：训练 ≤k-2、验证 k-1、测试 k，每折按训练年份重新标准化）。
另与滚动外推的 GBM(M1)/GBM(M4) 做秩平均集成，比较 "GBM(M4)+R-GCN" 与 "GBM(M1)+MLP"。"""
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import *

torch.set_num_threads(8)
M = importlib.import_module("04_model")
P = M.P
E_all = pd.read_parquet(WORK / "edges.parquet")
R_all = pd.read_parquet(WORK / "node_risk.parquet")
R_all = R_all.rename(columns={R_all.columns[0]: "node"}) if "node" not in R_all.columns else R_all
REL = M.REL
RGCN = importlib.import_module("07_gnn").RGCN
own_cols = [c for c in M.FIN + M.OWN_EV if c not in ("board_c", "sw_l1_c")]


def own_matrix(trm):
    Xo = P[own_cols].astype(float)
    lo, hi = Xo[trm].quantile(0.01), Xo[trm].quantile(0.99)
    Xo = Xo.clip(lo, hi, axis=1)
    mu, sd = Xo[trm].mean(), Xo[trm].std().replace(0, 1)
    na = Xo.isna().astype(float)
    na = na.loc[:, na[trm].mean() > 0.01].add_suffix("_na")
    Xo = ((Xo - mu) / sd).fillna(0)
    return pd.concat([Xo, na, pd.get_dummies(P.sw_l1, prefix="ind", dtype=float), pd.get_dummies(P.board, prefix="bd", dtype=float)], axis=1)


def build(t, Xown, noedge):
    E = E_all[E_all.t == t]
    foc = P[P.t == t]
    nodes = pd.Index(sorted(set(E.src) | set(E.dst) | set("C:" + foc.code)))
    N = len(nodes)
    ix = pd.Series(np.arange(N), index=nodes)
    rk = R_all[R_all.t == t].set_index("node").reindex(nodes).fillna(0)
    deg = pd.concat([E.src, E.dst]).value_counts().reindex(nodes).fillna(0).values / 2
    is_listed = nodes.str.startswith("C:")
    is_person = (~is_listed) & nodes.str[2:].str.fullmatch(r"[一-龥·]{2,4}")
    x_node = np.column_stack([np.log1p(rk[["fin", "trade", "viol", "dist"]].values), is_listed, is_person,
                              np.zeros(N) if noedge else np.log1p(deg)])
    x_own = np.zeros((N, Xown.shape[1]), dtype=np.float32)
    fidx = ix["C:" + foc.code].values
    x_own[fidx] = Xown.loc[foc.index].values
    is_focal = np.zeros(N)
    is_focal[fidx] = 1
    x = np.column_stack([x_own, x_node, is_focal]).astype(np.float32)
    adjs = []
    for r in REL:
        e = E[E.rel == r].iloc[:0] if noedge else E[E.rel == r]
        si, di = ix[e.src].values, ix[e.dst].values
        cnt = np.bincount(si, minlength=N).astype(np.float32)
        val = 1.0 / np.maximum(cnt[si], 1)
        adjs.append(torch.sparse_coo_tensor(np.vstack([si, di]), torch.tensor(val, dtype=torch.float32), (N, N)).coalesce())
    return {"x": torch.tensor(x), "adjs": adjs, "fidx": torch.tensor(fidx), "y": torch.tensor(foc.y.values, dtype=torch.float32),
            "index": foc.index.values}


def train_eval(train_years, val_year, test_year, noedge, seed=1):
    trm = P.t.isin(train_years)
    Xown = own_matrix(trm)
    graphs = {t: build(t, Xown, noedge) for t in list(train_years) + [val_year, test_year]}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RGCN(graphs[test_year]["x"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    pr = P[trm].y.mean()
    pw = torch.tensor((1 - pr) / pr * 0.3)
    best, state, bad = -1, None, 0
    for ep in range(200):
        model.train()
        for t in np.random.permutation(list(train_years)):
            g = graphs[t]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(g), g["y"], pos_weight=pw).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            a = roc_auc_score(graphs[val_year]["y"].numpy(), model(graphs[val_year]).numpy())
        if a > best:
            best, bad, state = a, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 30:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(graphs[test_year])).numpy()
    return pd.Series(p, index=graphs[test_year]["index"])


if __name__ == "__main__":
    oot = pd.read_parquet(WORK / "oot_rolling.parquet")
    out = []
    for k in [2019, 2020, 2021, 2022, 2023]:
        trn = [t for t in YEARS if t <= k - 2]
        pg = train_eval(trn, k - 1, k, noedge=False)
        pm = train_eval(trn, k - 1, k, noedge=True)
        d = P.loc[pg.index, ["code", "t", "y"]].copy()
        d["R-GCN"], d["MLP(同架构去边)"] = pg.values, pm.values
        d = d.merge(oot[["code", "t", "M1 自身", "M4 +结构位置"]], on=["code", "t"])
        rk = lambda s: s.rank(pct=True)
        d["GBM(M4)+R-GCN"] = 0.5 * rk(d["M4 +结构位置"]) + 0.5 * rk(d["R-GCN"])
        d["GBM(M1)+MLP"] = 0.5 * rk(d["M1 自身"]) + 0.5 * rk(d["MLP(同架构去边)"])
        out.append(d)
        print(k, {c: round(roc_auc_score(d.y, d[c]), 4) for c in ["M1 自身", "R-GCN", "MLP(同架构去边)", "GBM(M4)+R-GCN", "GBM(M1)+MLP"]}, flush=True)
    R = pd.concat(out, ignore_index=True)
    R.to_parquet(WORK / "gnn_rolling.parquet", index=False)
    y = R.y.values
    rng = np.random.default_rng(9)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    for a, b in [("R-GCN", "MLP(同架构去边)"), ("GBM(M4)+R-GCN", "GBM(M1)+MLP"), ("GBM(M1)+MLP", "M1 自身"), ("R-GCN", "M1 自身")]:
        dd = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            dd.append(roc_auc_score(y[idx], R[a].values[idx]) - roc_auc_score(y[idx], R[b].values[idx]))
        dd = np.array(dd)
        rows.append({"模型": a, "对照": b, "AUC": roc_auc_score(y, R[a]), "对照AUC": roc_auc_score(y, R[b]), "ΔAUC": dd.mean(),
                     "CI_lo": np.percentile(dd, 2.5), "CI_hi": np.percentile(dd, 97.5), "P(Δ≤0)": (dd <= 0).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gnn_rolling.csv", index=False, encoding="utf-8-sig")
    print(res.round(4).to_string(index=False))
