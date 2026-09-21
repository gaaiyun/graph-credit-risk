"""按观察年构建异质关联图，并生成 企业自身 / 一跳 / 多跳 / 结构 / 同业竞争 五组特征 + 标签。

时点口径（防穿越）：决策点 D_t=(t+1)-04-30。只用 FY t 年报（此前已披露）与公告日 ≤ D_t 的事件；
标签 = (D_t, D_t+12m] 内首次以被告身份披露金融债务类诉讼；过去 24 个月已发生过的企业不进样本。
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from common import *

master = pd.read_parquet(WORK / "firm_master.parquet")
sw = pd.read_parquet(WORK / "sw_hist.parquet")
fin = pd.read_parquet(WORK / "fin_panel.parquet")
cases = pd.read_parquet(WORK / "lit_cases.parquet")
parties = pd.read_parquet(WORK / "lit_parties.parquet")
vio = pd.read_parquet(WORK / "violations.parquet")
trades = pd.read_parquet(WORK / "trades.parquet")
inst = pd.read_parquet(WORK / "inst.parquet")
sc = pd.read_parquet(WORK / "sc_edges.parquet")
shr = pd.read_parquet(WORK / "shr_edges.parquet")
rp = pd.read_parquet(WORK / "rp_edges.parquet")

REL_TYPES = ["sup", "cus", "holder", "invest", "ctrl", "person", "affil", "rp_other", "codef", "debtor", "creditor"]
INVERSE = {"sup": "cus", "cus": "sup", "holder": "invest", "invest": "holder", "debtor": "creditor", "creditor": "debtor"}
HUB_DEG = 50           # 度数超过该值的外部实体不参与多跳传播，只保留一跳
ALPHA = 0.5            # 多跳衰减

own_fin = cases[cases.self_def & (cases.ctype == "fin")][["code", "date", "amt"]]
own_trade = cases[cases.self_def & (cases.ctype == "trade")][["code", "date", "amt"]]
own_other = cases[cases.self_def & (cases.ctype == "other")][["code", "date", "amt", "Cname"]]


def months_ago(dates, D):
    return (D - pd.to_datetime(dates)).dt.days / 30.4375


def in_window(dates, D, m_from, m_to=0):
    """日期落在 (D - m_from 个月, D - m_to 个月]"""
    lo, hi = D - pd.DateOffset(months=m_from), D - pd.DateOffset(months=m_to)
    return (dates > lo) & (dates <= hi)


def sw_at(D):
    s = sw[sw.start <= D].groupby("code").sw.last()
    return s


def node_of(code, name):
    return np.where(pd.notna(code) & (pd.Series(code).astype(str).str.len() == 6), "C:" + pd.Series(code).astype(str), "E:" + pd.Series(name).astype(str))


def build_year(t):
    D = decision_date(t)
    # ------------------------------------------------ 样本与标签
    ft = fin[fin.fy == t].copy()
    ind = sw_at(D)
    ft["sw"] = ft.code.map(ind)
    ft = ft[ft.sw.notna() & ~ft.sw.str[:2].isin(["48", "49"])]            # 剔除银行、非银金融
    ft = ft.merge(master[["code", "list_date", "board"]], on="code", how="left")
    ft = ft[ft.list_date.isna() | (ft.list_date < D)]
    recent = set(own_fin[in_window(own_fin.date, D, EXCLUDE_M)].code)
    ft = ft[~ft.code.isin(recent)].copy()
    fwd = own_fin[(own_fin.date > D) & (own_fin.date <= D + pd.DateOffset(months=12))]
    ft["y"] = ft.code.isin(set(fwd.code)).astype(int)
    fwd_big = fwd[fwd.amt >= 1000]
    ft["y_big"] = ft.code.isin(set(fwd_big.code)).astype(int)
    fwd_tr = own_trade[(own_trade.date > D) & (own_trade.date <= D + pd.DateOffset(months=12))]
    ft["y_broad"] = (ft.y.astype(bool) | ft.code.isin(set(fwd_tr.code))).astype(int)
    first_fwd = fwd.groupby("code").date.min()
    ft["event_date"] = ft.code.map(first_fwd)
    ft["t"] = t
    ft["firm_age"] = (D - ft.list_date).dt.days / 365.25
    ft["sw_l1"] = ft.sw.str[:2]
    ft["sw_l2"] = ft.sw.str[:4]
    focal = ft.code.tolist()

    # ------------------------------------------------ 企业自身事件特征（M1）
    def dsum(ev, key="code", m=LOOKBACK_M, lag=0):
        e = ev[in_window(ev.date, D, m, lag)]
        return pd.Series(decay(months_ago(e.date, D)), index=e.index).groupby(e[key].values).sum()

    ft["own_fin_old"] = ft.code.map(dsum(own_fin, m=LOOKBACK_M)).fillna(0)       # 只可能来自 24-36 个月前
    ft["own_trade"] = ft.code.map(dsum(own_trade)).fillna(0)
    ft["own_other_def"] = ft.code.map(dsum(own_other)).fillna(0)
    sec = own_other[own_other.Cname.fillna("").str.contains("虚假陈述")]
    ft["own_sec_fraud"] = ft.code.map(dsum(sec, m=24)).fillna(0)
    e12 = cases[cases.self_def & in_window(cases.date, D, 12)]
    ft["own_def_amt12"] = np.log1p(ft.code.map(e12.groupby("code").amt.sum()).fillna(0))
    ft["own_def_n12"] = ft.code.map(e12.groupby("code").size()).fillna(0)
    ov = vio[vio.Relationship == "公司本身"]
    ft["own_viol"] = ft.code.map(dsum(ov)).fillna(0)
    ft["own_viol_severe"] = ft.code.map(dsum(ov[ov.severe == 1])).fillna(0)
    inv = ov[ov.DispTp.fillna("").str.contains("立案调查")]
    ft["own_probe12"] = ft.code.map(inv[in_window(inv.date, D, 12)].groupby("code").size()).fillna(0)
    tr12 = trades[in_window(trades.date, D, 12)]
    it = inst[inst.fy == t].drop_duplicates("code").set_index("code")
    tot = ft.code.map(it.tot_shares)
    for tp, col in [("公司股东", "sell_major"), ("高管", "sell_exec"), ("个人股东", "sell_person")]:
        s = tr12[tr12.ShrhTp == tp].groupby("code").shares.sum()
        ft[col] = ft.code.map(s).fillna(0) / tot
    ft["inst_ratio"] = ft.code.map(it.inst_ratio)
    ft["fund_n"] = ft.code.map(it.fund_n)
    st = shr[shr.fy == t]
    top1 = st.sort_values("rank").drop_duplicates("code").set_index("code")
    ft["top1_stake"] = ft.code.map(top1.stake)
    ft["top1_person"] = ft.code.map(top1.person).astype(float)
    soe_pat = "国有资产|国资委|人民政府|财政|国有资本|国资|国务院|中央汇金|管理委员会|国家开发投资|中国.*集团有限公司$"
    ft["soe"] = ft.code.map(top1.name.str.contains(soe_pat)).astype(float)
    nonfin = st[~st.fin_holder]
    ft["top10_nonfin"] = ft.code.map(nonfin.groupby("code").stake.sum())

    # ------------------------------------------------ 节点风险（所有上市/非上市节点，36 个月衰减）
    ev = []
    for kind, e in [("fin", own_fin), ("trade", own_trade)]:
        e = e[in_window(e.date, D, LOOKBACK_M)]
        ev.append(pd.DataFrame({"node": "C:" + e.code, "w": decay(months_ago(e.date, D)), "kind": kind}))
    p = parties[(parties.side == "def") & parties.ctype.isin(["fin", "trade"]) & in_window(parties.date, D, LOOKBACK_M)]
    p = p[p.name_code.isna() | (p.name_code != p.code)]            # 披露方自身的被告记录已在上面计过
    ev.append(pd.DataFrame({"node": node_of(p.name_code.values, p.name.values), "w": decay(months_ago(p.date, D)),
                            "kind": p.ctype.values}))
    v = vio[in_window(vio.date, D, LOOKBACK_M)]
    v_own = v[v.Relationship == "公司本身"]
    ev.append(pd.DataFrame({"node": "C:" + v_own.code, "w": decay(months_ago(v_own.date, D)) * (1 + v_own.severe), "kind": "viol"}))
    v_sh = v[v.Relationship == "公司股东"].merge(pd.read_parquet(WORK / "name_dict.parquet").rename(columns={"name": "obj_name", "code": "oc"}),
                                             on="obj_name", how="left")
    ev.append(pd.DataFrame({"node": node_of(v_sh.oc.values, v_sh.obj_name.values), "w": decay(months_ago(v_sh.date, D)) * 2,
                            "kind": "viol"}))
    ev = pd.concat(ev, ignore_index=True)
    risk = ev.pivot_table(index="node", columns="kind", values="w", aggfunc="sum", fill_value=0)
    for k in ["fin", "trade", "viol"]:
        if k not in risk:
            risk[k] = 0.0
    # 上市公司节点附加财务困境信号（FY t 亏损、资不抵债）
    dist = fin[fin.fy == t].set_index("code")
    distress = ((dist.roa < 0).astype(float) + (dist.lev > 1).astype(float) + dist.loss_2y.fillna(0))
    risk = risk.reindex(risk.index.union("C:" + distress.index)).fillna(0)
    risk["dist"] = pd.Series(distress.values, index="C:" + distress.index).reindex(risk.index).fillna(0)
    risk["s"] = risk.fin + 0.5 * risk.trade + 0.25 * risk.viol          # 综合传播风险

    # ------------------------------------------------ 关系边（focal 视角 + 反向）
    E = []
    s1 = sc[sc.fy.isin([t, t - 1]) & ~sc.anon]
    w = s1.share.fillna(0.05).clip(0.005, 1) * np.where(s1.fy == t, 1.0, 0.5)
    E.append(pd.DataFrame({"src": "C:" + s1.code.values, "dst": node_of(s1.nb_code.values, s1.name.values), "rel": s1.rel.values, "w": w.values}))
    h = st[~st.fin_holder & (~st.person | (st.stake >= 5))]
    E.append(pd.DataFrame({"src": "C:" + h.code.values, "dst": node_of(h.name_code.values, h.name.values), "rel": "holder", "w": (h.stake / 100).values}))
    r = rp[in_window(rp.date, D, LOOKBACK_M)]
    r = r.drop_duplicates(["code", "name", "rtype"])
    E.append(pd.DataFrame({"src": "C:" + r.code.values, "dst": node_of(r.name_code.values, r.name.values),
                           "rel": np.where(r.rtype == "other", "rp_other", r.rtype), "w": 1.0}))
    cw = cases[in_window(cases.date, D, LOOKBACK_M)]
    pw = parties[(parties.side == "def") & in_window(parties.date, D, LOOKBACK_M)]
    pw = pw.merge(cw[["case_id", "self_def", "self_pla"]], on="case_id")
    cod = pw[pw.self_def & (pw.name_code.isna() | (pw.name_code != pw.code))]
    E.append(pd.DataFrame({"src": "C:" + cod.code.values, "dst": node_of(cod.name_code.values, cod.name.values), "rel": "codef", "w": 1.0}))
    deb = pw[pw.self_pla & pw.ctype.isin(["fin", "trade"])]
    E.append(pd.DataFrame({"src": "C:" + deb.code.values, "dst": node_of(deb.name_code.values, deb.name.values), "rel": "debtor", "w": 1.0}))
    E = pd.concat(E, ignore_index=True)
    E = E[(E.src != E.dst) & (E.dst.str.len() > 3)]
    E = E.groupby(["src", "dst", "rel"], as_index=False).w.max()
    rev = E.rename(columns={"src": "dst", "dst": "src"}).copy()
    rev["rel"] = rev.rel.map(lambda x: INVERSE.get(x, x))
    E = pd.concat([E, rev], ignore_index=True).groupby(["src", "dst", "rel"], as_index=False).w.max()

    nodes = pd.Index(sorted(set(E.src) | set(E.dst) | set("C:" + pd.Index(focal))))
    N = len(nodes)
    ix = pd.Series(np.arange(N), index=nodes)
    si, di = ix[E.src].values, ix[E.dst].values
    s_vec = risk.s.reindex(nodes).fillna(0).values
    fin_vec = risk.fin.reindex(nodes).fillna(0).values
    tr_vec = risk.trade.reindex(nodes).fillna(0).values
    dist_vec = risk.dist.reindex(nodes).fillna(0).values
    is_listed = nodes.str.startswith("C:")
    fidx = ix["C:" + pd.Index(focal)].values

    feats = {}
    W = {}
    for rel in REL_TYPES:
        m = (E.rel == rel).values
        A = sp.csr_matrix((E.w.values[m], (si[m], di[m])), shape=(N, N))
        B = sp.csr_matrix((np.ones(m.sum()), (si[m], di[m])), shape=(N, N))
        W[rel] = A
        feats[f"nb_{rel}_n"] = np.asarray(B[fidx].sum(1)).ravel()
        feats[f"nb_{rel}_fin"] = A[fidx] @ fin_vec
        feats[f"nb_{rel}_trade"] = A[fidx] @ tr_vec
        feats[f"nb_{rel}_risky"] = B[fidx] @ ((fin_vec + tr_vec) > 0).astype(float)
        feats[f"nb_{rel}_dist"] = B[fidx] @ (dist_vec * is_listed)

    # ------------------------------------------------ 多跳：对称归一化邻接（剔除枢纽实体），二跳去回声 + 三跳 PPR
    Abin = sp.csr_matrix((np.ones(len(si)), (si, di)), shape=(N, N))
    Abin.data[:] = 1.0
    Abin = ((Abin + Abin.T) > 0).astype(float).tocsr()
    deg = np.asarray(Abin.sum(1)).ravel()
    hub = (~is_listed) & (deg > HUB_DEG)
    keep = sp.diags((~hub).astype(float))
    Ah = keep @ Abin @ keep
    dh = np.asarray(Ah.sum(1)).ravel()
    dinv = sp.diags(np.where(dh > 0, 1 / np.sqrt(np.maximum(dh, 1e-12)), 0))
    An = (dinv @ Ah @ dinv).tocsr()
    As = An @ s_vec
    echo2 = np.asarray(An.multiply(An.T).sum(1)).ravel()          # (An^2)_ii
    for rel in REL_TYPES:
        Wr = W[rel]
        Wb = Wr.copy(); Wb.data[:] = 1.0
        Wb = keep @ Wb
        two = Wb[fidx] @ As - np.asarray(Wb[fidx].multiply(An.T[fidx]).sum(1)).ravel() * s_vec[fidx]
        feats[f"hop2_{rel}"] = two
    A2s = An @ As
    A3s = An @ A2s
    feats["hop2_all"] = A2s[fidx] - echo2[fidx] * s_vec[fidx]
    feats["ppr3"] = (As + ALPHA * (A2s - echo2 * s_vec) + ALPHA ** 2 * A3s)[fidx]
    feats["hop1_all"] = As[fidx]

    # ------------------------------------------------ 结构特征
    G = nx.Graph()
    G.add_nodes_from(range(N))
    ui, uj = Abin.nonzero()
    msk = ui < uj
    G.add_edges_from(zip(ui[msk], uj[msk]))
    pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-8)
    core = nx.core_number(G)
    feats["deg"] = deg[fidx]
    feats["deg_listed"] = np.asarray(Abin[fidx] @ is_listed.astype(float)).ravel()
    feats["pagerank"] = np.array([pr[i] for i in fidx]) * N
    feats["kcore"] = np.array([core[i] for i in fidx])
    # 集团/控制圈：持股≥20%的控股边 + 关联方披露的同一控制边 构成的连通分量（剔除枢纽与政府/国资委节点）
    ctrl_rel = ((E.rel.isin(["holder", "invest"]) & (E.w >= 0.2)) | (E.rel == "ctrl")).values
    gov = np.array([is_gov(n) for n in nodes.str[2:]])
    hub = hub | (gov & ~is_listed)
    Gc = nx.Graph()
    Gc.add_nodes_from(range(N))
    cs, cd = si[ctrl_rel], di[ctrl_rel]
    ok = ~hub[cs] & ~hub[cd]
    Gc.add_edges_from(zip(cs[ok], cd[ok]))
    comp_id = np.zeros(N, dtype=int)
    for k, comp in enumerate(nx.connected_components(Gc)):
        comp_id[list(comp)] = k
    comp_size = np.bincount(comp_id)
    comp_listed = np.bincount(comp_id, weights=is_listed.astype(float))
    comp_risk = np.bincount(comp_id, weights=s_vec)
    comp_fin = np.bincount(comp_id, weights=fin_vec)
    cid = comp_id[fidx]
    feats["group_size"] = comp_size[cid]
    feats["group_listed"] = comp_listed[cid]
    feats["group_risk"] = comp_risk[cid] - s_vec[fidx]
    feats["group_fin"] = comp_fin[cid] - fin_vec[fidx]

    F = pd.DataFrame(feats)
    F.insert(0, "code", focal)
    ft = ft.merge(F, on="code", how="left")

    # ------------------------------------------------ 同业竞争（申万二级，剔除自身）
    fall = fin[fin.fy == t].copy()
    fall["sw_l2"] = fall.code.map(ind).str[:4]
    fall = fall[fall.sw_l2.notna()]
    f12 = set(own_fin[in_window(own_fin.date, D, 12)].code)
    fall["fin12"] = fall.code.isin(f12)
    gp = fall.groupby("sw_l2")
    agg = pd.DataFrame({"n": gp.size(), "fin12": gp.fin12.sum(), "loss": gp.loss.sum(),
                        "roa_sum": gp.roa.sum(), "d_roa_sum": gp.d_roa.sum(), "rg_sum": gp.rev_growth.sum(),
                        "roa_n": gp.roa.count(), "d_roa_n": gp.d_roa.count(), "rg_n": gp.rev_growth.count()})
    a = agg.reindex(ft.sw_l2).reset_index(drop=True)
    me_fin = ft.code.isin(f12).astype(float).values
    n_ex = (a.n - 1).clip(lower=1)
    ft["peer_n"] = a.n.values - 1
    ft["peer_fin_rate"] = ((a.fin12 - me_fin) / n_ex).values
    ft["peer_loss_rate"] = ((a.loss - ft.loss.fillna(0).values) / n_ex).values

    def loo_mean(sum_col, n_col, own):
        o = own.values
        has = ~np.isnan(o)
        return np.where(has, (a[sum_col] - np.nan_to_num(o)) / (a[n_col] - 1).clip(lower=1), a[sum_col] / a[n_col].clip(lower=1))

    ft["peer_roa"] = loo_mean("roa_sum", "roa_n", ft.roa)
    ft["peer_d_roa"] = loo_mean("d_roa_sum", "d_roa_n", ft.d_roa)
    ft["peer_rev_growth"] = loo_mean("rg_sum", "rg_n", ft.rev_growth)
    ft["rel_roa"] = ft.roa - ft.peer_roa
    ft["rel_d_roa"] = ft.d_roa - ft.peer_d_roa
    ft["rel_rev_growth"] = ft.rev_growth - ft.peer_rev_growth
    print(f"[t={t}] D={D.date()} 样本 {len(ft)} 正例 {ft.y.sum()} ({ft.y.mean():.2%})  图: 节点 {N} 边 {len(E)//2} 枢纽 {hub.sum()}", flush=True)
    return ft, E.assign(t=t), risk.assign(t=t)


if __name__ == "__main__":
    panels, edges, risks = [], [], []
    for t in YEARS:
        ft, E, rk = build_year(t)
        panels.append(ft); edges.append(E); risks.append(rk.reset_index())
    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(WORK / "panel.parquet", index=False)
    pd.concat(edges, ignore_index=True).to_parquet(WORK / "edges.parquet", index=False)
    pd.concat(risks, ignore_index=True).to_parquet(WORK / "node_risk.parquet", index=False)
    print("panel", panel.shape, "正例总数", panel.y.sum())
