"""关系图卷积网络 R-GCN（纯 PyTorch 稀疏实现）。
每个观察年一张异质图（上市公司 + 外部实体，11 类关系），上市公司节点带自身特征，所有节点带风险事件特征。
训练 2016-2020 图，验证 2021 早停，测试 2022-2023；3 个种子平均。另给出 GBM(M4) 与 GNN 的秩平均集成。"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import importlib
from common import *

torch.set_num_threads(8)
M = importlib.import_module("04_model")
P = M.P
E_all = pd.read_parquet(WORK / "edges.parquet")
R_all = pd.read_parquet(WORK / "node_risk.parquet").rename(columns={"index": "node"})
if "node" not in R_all.columns:
    R_all = R_all.rename(columns={R_all.columns[0]: "node"})
REL = M.REL
NOEDGE = "mlp" in sys.argv          # 对照：同架构但不传消息、度数置零，等价于同容量 MLP
TAG = "gnn::MLP(同架构去边)" if NOEDGE else "gnn::R-GCN"

# 自身特征：训练期分位缩尾 + 标准化，缺失置 0 并加缺失指示
own_cols = [c for c in M.FIN + M.OWN_EV if c not in ("board_c", "sw_l1_c")]
Xo = P[own_cols].astype(float)
trm = P.t.isin(TRAIN_YEARS)
lo, hi = Xo[trm].quantile(0.01), Xo[trm].quantile(0.99)
Xo = Xo.clip(lo, hi, axis=1)
mu, sd = Xo[trm].mean(), Xo[trm].std().replace(0, 1)
na = Xo.isna().astype(float)
na = na.loc[:, na[trm].mean() > 0.01].add_suffix("_na")
Xo = ((Xo - mu) / sd).fillna(0)
ind = pd.get_dummies(P.sw_l1, prefix="ind", dtype=float)
brd = pd.get_dummies(P.board, prefix="bd", dtype=float)
Xown = pd.concat([Xo, na, ind, brd], axis=1)
D_OWN = Xown.shape[1]


def build_graph(t):
    E = E_all[E_all.t == t]
    foc = P[P.t == t]
    nodes = pd.Index(sorted(set(E.src) | set(E.dst) | set("C:" + foc.code)))
    N = len(nodes)
    ix = pd.Series(np.arange(N), index=nodes)
    rk = R_all[R_all.t == t].set_index("node").reindex(nodes).fillna(0)
    deg = pd.concat([E.src, E.dst]).value_counts().reindex(nodes).fillna(0).values / 2
    is_listed = nodes.str.startswith("C:")
    name = nodes.str[2:]
    is_person = (~is_listed) & name.str.fullmatch(r"[一-龥·]{2,4}")
    x_node = np.column_stack([np.log1p(rk[["fin", "trade", "viol", "dist"]].values), is_listed, is_person,
                              np.zeros(N) if NOEDGE else np.log1p(deg)])
    x_own = np.zeros((N, D_OWN), dtype=np.float32)
    fidx = ix["C:" + foc.code].values
    x_own[fidx] = Xown.loc[foc.index].values
    is_focal = np.zeros(N); is_focal[fidx] = 1
    x = np.column_stack([x_own, x_node, is_focal]).astype(np.float32)
    adjs = []
    for r in REL:
        e = E[E.rel == r].iloc[:0] if NOEDGE else E[E.rel == r]
        si, di = ix[e.src].values, ix[e.dst].values
        cnt = np.bincount(si, minlength=N).astype(np.float32)
        val = 1.0 / np.maximum(cnt[si], 1)                        # 按关系做行归一化（邻居均值）
        A = torch.sparse_coo_tensor(np.vstack([si, di]), torch.tensor(val, dtype=torch.float32), (N, N)).coalesce()
        adjs.append(A)
    return {"x": torch.tensor(x), "adjs": adjs, "fidx": torch.tensor(fidx), "y": torch.tensor(foc.y.values, dtype=torch.float32),
            "index": foc.index.values}


class RGCN(nn.Module):
    def __init__(self, d_in, d_h=64, n_rel=len(REL), dropout=0.3):
        super().__init__()
        self.inp = nn.Linear(d_in, d_h)
        self.rel1 = nn.ModuleList([nn.Linear(d_h, d_h, bias=False) for _ in range(n_rel)])
        self.self1 = nn.Linear(d_h, d_h)
        self.rel2 = nn.ModuleList([nn.Linear(d_h, d_h, bias=False) for _ in range(n_rel)])
        self.self2 = nn.Linear(d_h, d_h)
        self.out = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_h, 1))
        self.dp = dropout

    def forward(self, g):
        h0 = F.relu(self.inp(g["x"]))
        h = F.dropout(h0, self.dp, self.training)
        m = self.self1(h)
        for A, W in zip(g["adjs"], self.rel1):
            m = m + torch.sparse.mm(A, W(h))
        h = F.dropout(F.relu(m), self.dp, self.training)
        m = self.self2(h)
        for A, W in zip(g["adjs"], self.rel2):
            m = m + torch.sparse.mm(A, W(h))
        h = F.relu(m)
        z = torch.cat([h0, h], dim=1)[g["fidx"]]                  # 跳连：保留自身信息
        return self.out(z).squeeze(-1)


if __name__ == "__main__":
    graphs = {t: build_graph(t) for t in YEARS}
    print({t: (g["x"].shape[0], len(g["fidx"])) for t, g in graphs.items()}, flush=True)
    d_in = graphs[YEARS[0]]["x"].shape[1]
    pos_rate = P[P.t.isin(TRAIN_YEARS)].y.mean()
    preds = []
    for seed in [1, 2, 3]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = RGCN(d_in)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        pw = torch.tensor((1 - pos_rate) / pos_rate * 0.3)
        best, best_state, bad = -1, None, 0
        for ep in range(200):
            model.train()
            for t in np.random.permutation(TRAIN_YEARS):
                g = graphs[t]
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(g), g["y"], pos_weight=pw)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                gv = graphs[VALID_YEARS[0]]
                auc = roc_auc_score(gv["y"].numpy(), model(gv).numpy())
            if auc > best:
                best, bad = auc, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= 30:
                break
        model.load_state_dict(best_state)
        model.eval()
        p = pd.Series(np.nan, index=P.index)
        with torch.no_grad():
            for t, g in graphs.items():
                p.loc[g["index"]] = torch.sigmoid(model(g)).numpy()
        preds.append(p)
        te = P.t.isin(TEST_YEARS)
        print(f"seed {seed}: epochs={ep + 1} best valid AUC={best:.4f} test AUC={roc_auc_score(P.y[te], p[te]):.4f}", flush=True)
    gnn = pd.concat(preds, axis=1).mean(axis=1)
    pred = pd.read_parquet(WORK / "predictions.parquet")
    pred[TAG] = gnn.values
    te = P.t.isin(TEST_YEARS).values
    if not NOEDGE:
        rk = lambda s: pd.Series(s).rank(pct=True).values
        ens = np.full(len(P), np.nan)
        for t in P.t.unique():
            m = (P.t == t).values
            ens[m] = 0.5 * rk(pred.loc[m, "gbm::M4 +结构位置"].values) + 0.5 * rk(gnn.values[m])
        pred["ens::GBM(M4)+R-GCN"] = ens
    pred.to_parquet(WORK / "predictions.parquet", index=False)
    yt = P.y.values[te]
    rows = []
    for c in ["gbm::M1 自身", "gbm::M4 +结构位置", "gnn::R-GCN", "gnn::MLP(同架构去边)", "ens::GBM(M4)+R-GCN"]:
        if c in pred:
            rows.append({"模型": c, **M.metrics(yt, pred.loc[te, c].values)})
    res = pd.DataFrame(rows)
    if "gnn::MLP(同架构去边)" in pred:
        rk2 = lambda s: pd.Series(s).rank(pct=True).values
        ens0 = np.full(len(P), np.nan)
        for t in P.t.unique():
            m = (P.t == t).values
            ens0[m] = 0.5 * rk2(pred.loc[m, "gbm::M1 自身"].values) + 0.5 * rk2(pred.loc[m, "gnn::MLP(同架构去边)"].values)
        pred["ens::GBM(M1)+MLP"] = ens0
        pred.to_parquet(WORK / "predictions.parquet", index=False)
        rows.append({"模型": "ens::GBM(M1)+MLP", **M.metrics(yt, pred.loc[te, "ens::GBM(M1)+MLP"].values)})
        res = pd.DataFrame(rows)
    if "gnn::R-GCN" in pred and "gnn::MLP(同架构去边)" in pred:
        for a, b in [("gnn::R-GCN", "gnn::MLP(同架构去边)"), ("ens::GBM(M4)+R-GCN", "ens::GBM(M1)+MLP")]:
            rng = np.random.default_rng(5)
            pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
            d = []
            for _ in range(2000):
                idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                d.append(roc_auc_score(yt[idx], pred.loc[te, a].values[idx]) - roc_auc_score(yt[idx], pred.loc[te, b].values[idx]))
            d = np.array(d)
            res.loc[res.模型 == a, "对照"] = b
            res.loc[res.模型 == a, "ΔAUC"] = d.mean()
            res.loc[res.模型 == a, "CI_lo"] = np.percentile(d, 2.5)
            res.loc[res.模型 == a, "CI_hi"] = np.percentile(d, 97.5)
    if False:
        rng = np.random.default_rng(5)
        pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
        a_ = pred.loc[te, "gnn::R-GCN"].values
        b_ = pred.loc[te, "gnn::MLP(同架构去边)"].values
        d = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            d.append(roc_auc_score(yt[idx], a_[idx]) - roc_auc_score(yt[idx], b_[idx]))
        d = np.array(d)
        res["R-GCN减去MLP的ΔAUC"] = [np.nan] * len(res)
        res.loc[res.模型 == "gnn::R-GCN", "R-GCN减去MLP的ΔAUC"] = d.mean()
        res.loc[res.模型 == "gnn::R-GCN", "CI_lo"] = np.percentile(d, 2.5)
        res.loc[res.模型 == "gnn::R-GCN", "CI_hi"] = np.percentile(d, 97.5)
    res.to_csv(OUT / "gnn_compare.csv", index=False, encoding="utf-8-sig")
    print(res.round(4).to_string(index=False))
