"""月度动态面板：企业-月，评分日 S = 观察年 t 的第 k 个月末（k=0..11，自 D_t 所在的 4 月起）。
样本：年度样本中在 S 之前尚未发生首次违约的企业；标签：(S, S+6m] / (S, S+12m] 内首次金融债务违约。
特征（全部只用 S 及以前的信息）：
  自身动态：自身各类事件近 3/6/12 月计数、最新季报资产负债结构、评分月序
  关联事件时序：10 类一跳关系 × (近1月/近3月/近12月计数、3 月半衰期衰减值)，按事件类型再汇总（金融/欠款/执行失信）
  多层级：共同控股股东或同一控制下的兄弟上市公司、股权投资方（创投/产业基金）的其他被投上市公司、
          经客户或供应商的二跳主体、同业（申万二级）近 3/6 月违约数
剔除企业自身作为当事人的案件，避免把同一宗违约诉讼当成关联信号。"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
EV = pd.read_parquet(WORK / "events_all.parquet")
pc = pd.read_parquet(WORK / "party_cases.parquet")
cases = pd.read_parquet(WORK / "lit_cases.parquet")
shr = pd.read_parquet(WORK / "shr_edges.parquet")
sw = pd.read_parquet(WORK / "sw_hist.parquet")
trades = pd.read_parquet(WORK / "trades.parquet")
inst = pd.read_parquet(WORK / "inst.parquet")

REL = ["sup", "cus", "holder", "invest", "ctrl", "person", "affil", "rp_other", "codef", "debtor"]
RISK_KINDS = ["fin", "trade", "enforce", "shixin"]
HL = 3.0              # 动态事件半衰期（月）
PAST = 24             # 事件回看月数
own_fin = cases[cases.self_def & (cases.ctype == "fin")][["code", "date"]]
own_fin_mi = own_fin.assign(mi=own_fin.date.dt.year * 12 + own_fin.date.dt.month - 1)
pc_key = set(zip(pc.case_id.values, ("C:" + pc.code.astype(str)).values))
PE_PAT = re.compile(r"创业投资|创新投资|股权投资|产业投资|投资中心|投资基金|资本管理|合伙企业")

# 季报资产负债结构（按法定披露截止日可得）
sol = load_raw("偿债能力", 1)
sol["code"] = sol.Scode.str.zfill(6)
sol["qend"] = pd.to_datetime(sol.Date, errors="coerce")
lag = {3: 1, 6: 2, 9: 1, 12: 4}                     # 季末到披露截止的月数：一季报4/30、半年报8/31、三季报10/31、年报次年4/30
sol["avail_mi"] = sol.qend.dt.year * 12 + sol.qend.dt.month - 1 + sol.qend.dt.month.map(lag)
for src, dst in [("Aslbrt", "q_lev"), ("Curtrt", "q_cur"), ("Qikrt", "q_quick"), ("Cashrt", "q_cash"), ("Lwirt", "q_intdebt")]:
    sol[dst] = num(sol[src])
QCOLS = ["q_lev", "q_cur", "q_quick", "q_cash", "q_intdebt"]
sol = sol[["code", "avail_mi"] + QCOLS].dropna(subset=["avail_mi"])
sol["avail_mi"] = sol.avail_mi.astype("int64")
sol = sol.sort_values("avail_mi")


def window_feats(arr, m_s, prefix):
    """arr: [n, M] 月度计数；返回 m_s 时刻的近 1/3/12 月计数与半衰期衰减值"""
    out = {}
    for w in (1, 3, 12):
        out[f"{prefix}_{w}m"] = arr[:, max(0, m_s - w + 1): m_s + 1].sum(1)
    ages = m_s - np.arange(0, m_s + 1)
    out[f"{prefix}_dec"] = (arr[:, : m_s + 1] * np.power(0.5, ages / HL)).sum(1)
    return out


def build_cohort(t):
    D = decision_date(t)
    s0 = D.year * 12 + D.month - 1                  # D_t 所在月
    base = s0 - PAST                                # 数组第 0 列对应的月份
    M = PAST + 12
    foc = P[P.t == t].reset_index(drop=True)
    codes = foc.code.values
    fi = pd.Series(np.arange(len(codes)), index="C:" + codes)
    nF = len(codes)
    ev = EV[(EV.mi >= base) & (EV.mi < base + M)]
    ev_r = ev[ev.kind.isin(RISK_KINDS)]

    # ---------- 一跳关联事件：A[f, rel, m]
    Et = E[(E.t == t) & E.src.isin(fi.index) & E.rel.isin(REL)][["src", "dst", "rel"]].drop_duplicates()
    J = Et.merge(ev_r, left_on="dst", right_on="node")
    J = J[[(c, s) not in pc_key for c, s in zip(J.case_id.values, J.src.values)]]
    A = np.zeros((nF, len(REL), M), dtype=np.float32)
    np.add.at(A, (fi[J.src].values, pd.Index(REL).get_indexer(J.rel), (J.mi - base).values), 1.0)
    Ak = {}
    for kind_group, kinds in [("fin", ["fin"]), ("trade", ["trade"]), ("enf", ["enforce", "shixin"])]:
        Jk = J[J.kind.isin(kinds)].drop_duplicates(["src", "dst", "case_id", "mi"])
        a = np.zeros((nF, M), dtype=np.float32)
        np.add.at(a, (fi[Jk.src].values, (Jk.mi - base).values), 1.0)
        Ak[kind_group] = a

    # ---------- 兄弟企业：共同控股股东/同一控制（sib_ctrl）、共同创投或产业基金（sib_pe）
    st = shr[shr.fy == t]
    ctrl_h = st[(~st.fin_holder) & (st.stake >= 5)]
    ctrl_h = ctrl_h[~ctrl_h.name.map(is_gov)]
    pe_h = st[st.name.str.contains(PE_PAT) & (st.stake >= 1)]
    sib = {}
    for key, h in [("sib_ctrl", ctrl_h), ("sib_pe", pe_h)]:
        pairs = h[["name", "code"]].merge(h[["name", "code"]], on="name", suffixes=("", "_k"))
        pairs = pairs[pairs.code != pairs.code_k][["code", "code_k"]]
        if key == "sib_ctrl":
            ct = E[(E.t == t) & (E.rel == "ctrl") & E.src.isin(fi.index) & E.dst.str.startswith("C:")]
            pairs = pd.concat([pairs, pd.DataFrame({"code": ct.src.str[2:], "code_k": ct.dst.str[2:]})])
        pairs = pairs.drop_duplicates()
        pairs = pairs[pairs.code.isin(codes)]
        Js = pairs.assign(node="C:" + pairs.code_k).merge(ev_r, on="node")
        Js = Js[[(c, "C:" + s) not in pc_key for c, s in zip(Js.case_id.values, Js.code.values)]]
        a = np.zeros((nF, M), dtype=np.float32)
        np.add.at(a, (fi["C:" + Js.code].values, (Js.mi - base).values), 1.0)
        sib[key] = a
        sib[key + "_n"] = pairs.groupby("code").size().reindex(codes).fillna(0).values

    # ---------- 经客户/供应商的二跳（剔除枢纽中介与回到自身的路径）
    bus = E[(E.t == t) & E.rel.isin(["sup", "cus"])][["src", "dst"]].drop_duplicates()
    deg = pd.concat([E[E.t == t].src, E[E.t == t].dst]).value_counts()
    bus = bus[bus.dst.map(deg).fillna(0) <= 100]
    two = bus[bus.src.isin(fi.index)].merge(bus.rename(columns={"src": "mid", "dst": "far"}), left_on="dst", right_on="mid")
    two = two[two.far != two.src][["src", "far"]].drop_duplicates()
    Jt = two.merge(ev_r, left_on="far", right_on="node")
    Jt = Jt[[(c, s) not in pc_key for c, s in zip(Jt.case_id.values, Jt.src.values)]]
    A2 = np.zeros((nF, M), dtype=np.float32)
    np.add.at(A2, (fi[Jt.src].values, (Jt.mi - base).values), 1.0)

    # ---------- 同业（申万二级）违约动态
    ind = sw[sw.start <= D].groupby("code").sw.last().str[:4]
    of = own_fin_mi[(own_fin_mi.mi >= base) & (own_fin_mi.mi < base + M)].copy()
    of["l2"] = of.code.map(ind)
    of = of.dropna(subset=["l2"]).drop_duplicates(["code", "mi"])
    l2_idx = {g: i for i, g in enumerate(sorted(ind.dropna().unique()))}
    PL = np.zeros((len(l2_idx), M), dtype=np.float32)
    np.add.at(PL, (of.l2.map(l2_idx).values, (of.mi - base).values), 1.0)
    my_l2 = foc.sw_l2.map(l2_idx).fillna(-1).astype(int).values

    # ---------- 自身事件
    ev_own = ev[ev.node.isin(fi.index)]
    O = {}
    for kg, kinds in [("own_trade", ["trade"]), ("own_enf", ["enforce", "shixin"]), ("own_viol", ["viol"])]:
        e = ev_own[ev_own.kind.isin(kinds)].drop_duplicates(["node", "case_id", "mi", "kind"])
        a = np.zeros((nF, M), dtype=np.float32)
        np.add.at(a, (fi[e.node].values, (e.mi - base).values), 1.0)
        O[kg] = a
    tr = trades.assign(mi=trades.date.dt.year * 12 + trades.date.dt.month - 1)
    tr = tr[tr.code.isin(codes) & (tr.mi >= base) & (tr.mi < base + M) & (tr.ShrhTp == "公司股东")]
    SEL = np.zeros((nF, M), dtype=np.float32)
    np.add.at(SEL, (fi["C:" + tr.code].values, (tr.mi - base).values), tr.shares.fillna(0).values.astype(np.float32))
    tot = foc.code.map(inst[inst.fy == t].drop_duplicates("code").set_index("code").tot_shares).values

    # ---------- 首次违约时间
    fut = own_fin_mi[own_fin_mi.code.isin(codes) & (own_fin_mi.date > D)].groupby("code").date.min()
    first_def = foc.code.map(fut)

    out = []
    for k in range(12):
        m_s = PAST + k
        S = (D + pd.offsets.MonthEnd(k)) if k > 0 else D
        alive = first_def.isna() | (first_def > S)
        f = {"code": codes, "t": t, "k": k, "S": S}
        f["y6"] = (first_def > S) & (first_def <= S + pd.DateOffset(months=6))
        f["y12"] = (first_def > S) & (first_def <= S + pd.DateOffset(months=12))
        f["def_date"] = first_def.values
        for ri, r in enumerate(REL):
            f.update(window_feats(A[:, ri, :], m_s, f"dy_{r}"))
        for kg, a in Ak.items():
            f.update(window_feats(a, m_s, f"dy_all_{kg}"))
        for key in ["sib_ctrl", "sib_pe"]:
            f.update(window_feats(sib[key], m_s, f"dy_{key}"))
            f[f"{key}_n"] = sib[key + "_n"]
        f.update(window_feats(A2, m_s, "dy_sc2hop"))
        pl3 = PL[:, m_s - 2: m_s + 1].sum(1)
        pl6 = PL[:, m_s - 5: m_s + 1].sum(1)
        f["dy_peer_def_3m"] = np.where(my_l2 >= 0, pl3[np.maximum(my_l2, 0)], 0)
        f["dy_peer_def_6m"] = np.where(my_l2 >= 0, pl6[np.maximum(my_l2, 0)], 0)
        for kg, a in O.items():
            f.update(window_feats(a, m_s, kg))
        f["own_sell_6m"] = SEL[:, m_s - 5: m_s + 1].sum(1) / tot
        df = pd.DataFrame(f)[alive.values]
        out.append(df)
    C = pd.concat(out, ignore_index=True)
    # 最新可得季报
    C["mi_S"] = (C.S.dt.year * 12 + C.S.dt.month - 1).astype("int64")
    C = C.sort_values("mi_S")
    q = sol[sol.code.isin(codes)].rename(columns={"avail_mi": "mi_S"})
    C = pd.merge_asof(C, q.sort_values("mi_S"), on="mi_S", by="code", direction="backward")
    print(f"[t={t}] 企业-月 {len(C):,}  y6 正例 {int(C.y6.sum())}  关联事件匹配 {len(J):,}  兄弟对 ctrl/pe "
          f"{int((sib['sib_ctrl_n'] > 0).sum())}/{int((sib['sib_pe_n'] > 0).sum())}  二跳路径 {len(two):,}", flush=True)
    return C


if __name__ == "__main__":
    parts = [build_cohort(t) for t in YEARS]
    DY = pd.concat(parts, ignore_index=True)
    fcols = [c for c in DY.columns if c.startswith(("dy_", "own_", "sib_", "q_"))]
    DY[fcols] = DY[fcols].astype("float32")
    DY["y6"] = DY.y6.astype(int)
    DY["y12"] = DY.y12.astype(int)
    DY.to_parquet(WORK / "dyn_panel.parquet", index=False)
    print("动态面板", DY.shape, "y6 正例", int(DY.y6.sum()), "企业-年违约数", DY[DY.y6 == 1].groupby(["code", "t"]).ngroups)
