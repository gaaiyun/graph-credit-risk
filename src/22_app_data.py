"""产品原型数据导出（全部脱敏）→ docs/app.html。
样本企业（测试期 2019-2023 滚动外推）：被提前预警的违约、漏检的违约、预警未违约（误报）、正常，四类都展示。
每家：12 个月的动态/自身模型风险分位与校准后违约概率、关联事件时间轴、多层级关联网络、逐月 SHAP 归因、叙述。
另导出：预警工作台（每个企业-观察年首次进入前 5% 的月份）、事件乘数、证据面板。"""
import secrets
import json
import re
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import importlib
from common import *

M19 = importlib.import_module("19_dynamic_model")
DY, SETS = M19.DY, M19.SETS
names = list(SETS)
N_D1, N_D4 = names[0], names[3]
F4 = SETS[N_D4]
R = pd.read_parquet(WORK / "dyn_oot.parquet")
EV = pd.read_parquet(WORK / "events_all.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
P = pd.read_parquet(WORK / "panel.parquet")
pc = pd.read_parquet(WORK / "party_cases.parquet")
shr = pd.read_parquet(WORK / "shr_edges.parquet")
pc_key = set(zip(pc.case_id.values, ("C:" + pc.code.astype(str)).values))
ROOT_DIR = Path(__file__).resolve().parents[1]
NEG_RATE = 0.3                                                # 训练负样本抽样比例 → 校准几率
KIND_CN = {"fin": "金融债务被诉", "trade": "经营欠款被诉", "enforce": "被强制执行/查封冻结", "shixin": "失信/限制高消费"}
REL_CN = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员",
          "affil": "联营/合营/子公司", "rp_other": "其他关联方", "codef": "共同被告", "debtor": "债务人", "creditor": "债权人"}
LANE = {"holder": "股权", "invest": "股权", "ctrl": "股权", "person": "股权", "affil": "股权", "rp_other": "股权",
        "codef": "担保连带", "sup": "交易", "cus": "交易", "debtor": "交易", "sib_ctrl": "兄弟企业", "sib_pe": "兄弟企业", "sc2hop": "二跳"}

# ---------------------------------------------------------------- 分位与校准
for n, k in [(N_D1, "p1"), (N_D4, "p4")]:
    R[k] = R.groupby("S")[n].rank(pct=True)
    o = R[n] / (1 - R[n]) * NEG_RATE
    R["pd" + k[1]] = o / (1 + o)
R["ym"] = R.S.dt.strftime("%Y-%m")

# ---------------------------------------------------------------- SHAP 分组
M4 = importlib.import_module("04_model")
G_OWN_FIN = set(M4.FIN) | {c for c in F4 if c.startswith(("q_", "dq_"))}
G_OWN_EV = (set(M4.OWN_EV) | set(M19.OWN_DYN)) - G_OWN_FIN
G_STATIC = set(M4.HOP1 + M4.MULTI + M4.STRUCT)
G_DYN1 = set(M19.DYN1)
G_DYNM = set(M19.DYNM)
GROUPS = [("自身财务", G_OWN_FIN), ("自身事件与治理", G_OWN_EV), ("静态图谱结构", G_STATIC), ("一跳关联事件时序", G_DYN1), ("多层级时序", G_DYNM)]
W_CN = {"1m": "近1月", "3m": "近3月", "12m": "近12月", "dec": "衰减"}
BASE_CN = {"impair_to_rev": "资产减值/营收", "lev": "资产负债率", "roa": "ROA", "cfo_to_debt": "现金债务总额比", "cash_ratio": "现金比率",
           "cur_ratio": "流动比率", "quick_ratio": "速动比率", "log_ta": "总资产(对数)", "fin_exp_ratio": "财务费用率", "wc_to_loan": "营运资金/借款",
           "d_roa": "ROA变动", "tang_lev": "有形资产负债率", "cfo_to_due": "现金流到期债务保障", "firm_age": "上市年限", "fund_n": "持股基金家数",
           "top1_stake": "第一大股东持股", "inst_ratio": "机构持股比例", "sw_l1_c": "所属行业", "board_c": "上市板块", "int_debt_ratio": "带息负债比率",
           "d_cfo_to_debt": "现金债务比变动", "period_exp_ratio": "期间费用率", "own_viol": "违规处罚(年度)", "own_trade": "欠款被诉(年度)",
           "own_other_def": "其他被诉(年度)", "k": "距年报月数", "own_sell_6m": "大股东近6月净减持", "pagerank": "关联网络中心度", "deg": "关联度数",
           "rel_rev_growth": "相对同业营收增速", "gross_margin": "毛利率", "log_rev": "营业收入(对数)", "sell_exec": "高管净减持", "soe": "国有控股",
           "sib_ctrl_n": "兄弟企业数(共同控股股东)", "sib_pe_n": "其他被投企业数(共同创投)", "dy_peer_def_3m": "同业近3月违约数", "dy_peer_def_6m": "同业近6月违约数"}
QCN = {"q_lev": "最新季报资产负债率", "q_cur": "最新季报流动比率", "q_quick": "最新季报速动比率", "q_cash": "最新季报现金比率", "q_intdebt": "最新季报带息负债比",
       "dq_lev": "季报负债率较年报变化", "dq_cur_ratio": "季报流动比率较年报变化", "dq_quick_ratio": "季报速动比率较年报变化",
       "dq_cash_ratio": "季报现金比率较年报变化", "dq_int_debt_ratio": "季报带息负债较年报变化"}


def fname(c):
    if c in BASE_CN:
        return BASE_CN[c]
    if c in QCN:
        return QCN[c]
    m = re.fullmatch(r"own_(trade|enf|viol)_(1m|3m|12m|dec)", c)
    if m:
        return {"trade": "自身欠款被诉", "enf": "自身被强制执行", "viol": "自身违规处罚"}[m.group(1)] + f"({W_CN[m.group(2)]})"
    m = re.fullmatch(r"dy_(.+)_(1m|3m|12m|dec)", c)
    if m:
        k = m.group(1)
        lab = {"all_fin": "关联方金融债务被诉", "all_trade": "关联方欠款被诉", "all_enf": "关联方强制执行/失信", "sib_ctrl": "兄弟企业(共同控股股东)事件",
               "sib_pe": "其他被投企业(共同创投)事件", "sc2hop": "客户的客户/供应商的供应商事件"}.get(k, REL_CN.get(k, k) + "事件")
        return f"{lab}({W_CN[m.group(2)]})"
    m = re.fullmatch(r"nb_(\w+?)_(n|fin|trade|risky|dist)", c)
    if m and m.group(1) in REL_CN:
        return f"{REL_CN[m.group(1)]}·" + {"n": "数量", "fin": "金融违约暴露", "trade": "欠款暴露", "risky": "风险邻居数", "dist": "财务困境邻居"}[m.group(2)]
    if c.startswith("hop2_"):
        return f"二跳风险·{REL_CN.get(c[5:], c[5:])}"
    return {"ppr3": "三跳扩散风险", "hop1_all": "一跳风险合计", "group_fin": "控制圈违约暴露", "group_size": "控制圈规模"}.get(c, FEAT_CN.get(c, c))


boosters = {k: lgb.Booster(model_file=str(WORK / f"dyn_D4_fold{k}.txt")) for k in [2019, 2020, 2021, 2022, 2023]}


def shap_rows(idx):
    """idx: DY 行号；按观察年用对应折的 D4 模型计算 SHAP（对数几率尺度）"""
    out = {}
    for t, ix in pd.Series(idx, index=idx).groupby(DY.loc[idx, "t"].values):
        X = DY.loc[ix.values, F4].to_numpy(np.float32)
        contrib = boosters[t].predict(X, pred_contrib=True)[:, :-1]
        for r, row in zip(ix.values, contrib):
            out[r] = row
    return out


_ANON = {}


def anon(code):
    """每次构建随机生成、不落盘的匿名编号：公开的代码无法反推股票代码"""
    if code not in _ANON:
        used = set(_ANON.values())
        while True:
            v = "企业 #" + secrets.token_hex(2).upper()
            if v not in used:
                break
        _ANON[code] = v
    return _ANON[code]


# ---------------------------------------------------------------- 选样本企业
R["def_after_S"] = R.def_date.notna() & (R.def_date > R.S)
coh = R.groupby(["code", "t"]).agg(maxp4=("p4", "max"), maxp1=("p1", "max"), def_date=("def_date", "first"), S0=("S", "min"), S11=("S", "max"))
coh["defaulted"] = coh.def_date.notna() & (coh.def_date <= coh.S11 + pd.DateOffset(months=6))
pre = R[R.def_after_S & (R.def_date <= R.S + pd.DateOffset(months=6))].groupby(["code", "t"]).agg(pre_p4=("p4", "max"), pre_p1=("p1", "max"))
coh = coh.join(pre)
dyc = DY.set_index(["code", "t", "k"])
evsum_cols = [c for c in DY.columns if c.startswith("dy_") and c.endswith("_3m") and not c.startswith(("dy_all_", "dy_peer_"))]
R = R.merge(DY[["code", "t", "k"] + evsum_cols], on=["code", "t", "k"], how="left")
R["ev3"] = R[evsum_cols].fillna(0).sum(1)
coh = coh.join(R.groupby(["code", "t"]).ev3.max().rename("maxev3"))
coh["gain"] = coh.pre_p4 - coh.pre_p1
coh["def_mm"] = (coh.def_date.dt.year * 12 + coh.def_date.dt.month - 1) - (coh.S0.dt.year * 12 + coh.S0.dt.month - 1)
room = coh.def_mm.isna() | (coh.def_mm >= 4)
rs = np.random.default_rng(3)
caught = coh[room & coh.defaulted & (coh.pre_p4 >= 0.95) & (coh.maxev3 > 0)].sort_values(["gain", "pre_p4"], ascending=False).head(14)
missed = coh[room & coh.defaulted & (coh.pre_p4 < 0.8)].sort_values("maxev3", ascending=False).head(4)
fp = coh[~coh.defaulted & coh.def_date.isna() & (coh.maxp4 >= 0.97) & (coh.maxev3 > 0)].sort_values("maxp4", ascending=False).head(6)
okc = coh[coh.def_date.isna() & (coh.maxp4 < 0.5)]
okc = okc.iloc[rs.choice(len(okc), 8, replace=False)] if len(okc) > 8 else okc
sel = pd.concat([caught.assign(group="def"), missed.assign(group="miss"), fp.assign(group="fp"), okc.assign(group="ok")])
sel = sel[~sel.index.duplicated()]
print("样本企业", sel.group.value_counts().to_dict())
firm_id = {ix: f"A{i + 1:02d}" for i, ix in enumerate(sel.index)}
ind_of = P.set_index(["code", "t"]).sw_l1.map(SW_L1)

# ---------------------------------------------------------------- 每家企业
REL1 = ["holder", "codef", "debtor", "cus", "sup", "ctrl", "affil", "person", "invest", "rp_other"]


def cohort_events(code, t, D):
    """该企业在观察期(及前 24 个月)的关联事件：一跳、兄弟企业、二跳；剔除本企业为当事人的案件"""
    s0 = D.year * 12 + D.month - 1
    lo, hi = s0 - 24, s0 + 11
    ev = EV[(EV.mi >= lo) & (EV.mi <= hi) & EV.kind.isin(list(KIND_CN))]
    focal = "C:" + code
    rows = []
    one = E[(E.t == t) & (E.src == focal) & E.rel.isin(REL1)][["dst", "rel"]].drop_duplicates("dst")
    j = one.merge(ev, left_on="dst", right_on="node")
    rows.append(j.assign(hop=1, via=None))
    st = shr[shr.fy == t]
    ctrl_h = st[(~st.fin_holder) & (st.stake >= 5) & ~st.name.map(is_gov)]
    pe_h = st[st.name.str.contains(M19_PE) & (st.stake >= 1)]
    for key, h in [("sib_ctrl", ctrl_h), ("sib_pe", pe_h)]:
        mine = h[h.code == code].name.unique()
        sibs = h[h.name.isin(mine) & (h.code != code)][["name", "code"]].drop_duplicates("code")
        js = sibs.assign(dst="C:" + sibs.code).merge(ev, left_on="dst", right_on="node")
        rows.append(js.assign(rel=key, hop=2, via=np.where(js.name.map(is_person), "自然人股东", js.name)))
    bus = E[(E.t == t) & E.rel.isin(["sup", "cus"])][["src", "dst", "rel"]]
    mid = bus[bus.src == focal]
    two = mid.merge(bus.rename(columns={"src": "mid", "dst": "far", "rel": "rel2"}), left_on="dst", right_on="mid")
    two = two[two.far != focal].drop_duplicates("far")
    jt = two.merge(ev, left_on="far", right_on="node")
    rows.append(pd.DataFrame({"dst": jt.far, "rel": "sc2hop", "node": jt.node, "date": jt.date, "kind": jt.kind, "case_id": jt.case_id, "mi": jt.mi,
                              "hop": 2, "via": jt.mid}))
    X = pd.concat(rows, ignore_index=True)
    X = X.loc[np.array([(c, focal) not in pc_key for c in X.case_id.values], dtype=bool)]
    X["m"] = X.mi - s0
    return X.drop_duplicates(["dst", "case_id", "kind", "mi"]), one


M19_PE = M19_PE_PAT = __import__("re").compile(r"创业投资|创新投资|股权投资|产业投资|投资中心|投资基金|资本管理|合伙企业")
firms = []
for (code, t), row in sel.iterrows():
    D = decision_date(t)
    X, one = cohort_events(code, t, D)
    rr = R[(R.code == code) & (R.t == t)].sort_values("k")
    idx = DY.index[(DY.code.values == code) & (DY.t.values == t)]
    shp = shap_rows(idx)
    kmap = dict(zip(DY.loc[idx, "k"].values, idx))
    # 节点与匿名标签
    labels, cnt = {}, {}

    def lab(node, rel):
        if node in labels:
            return labels[node]
        name = node[2:]
        base = ("上市公司" if node.startswith("C:") else ("自然人" if is_person(name) else REL_CN.get(rel, {"sib_ctrl": "兄弟企业", "sib_pe": "其他被投企业",
                                                                                                                  "sc2hop": "二跳主体"}.get(rel, "关联方"))))
        if rel in ("sib_ctrl", "sib_pe"):
            base = "兄弟企业" if rel == "sib_ctrl" else "其他被投企业"
        if rel == "sc2hop":
            base = "二跳主体"
        cnt[base] = cnt.get(base, 0) + 1
        labels[node] = f"{base}{cnt[base]}"
        return labels[node]

    risky_nodes = X.groupby("dst").size().sort_values(ascending=False)
    n1 = [(d, r) for d, r in zip(one.dst, one.rel)]
    risky1 = [(d, r) for d, r in n1 if d in risky_nodes.index][:12]
    others = [(d, r) for d, r in n1 if d not in risky_nodes.index and r in ("holder", "cus", "sup", "codef", "ctrl")][: max(0, 14 - len(risky1))]
    ring1 = risky1 + others
    ring2 = [(d, r, v) for d, r, v in X[X.hop == 2][["dst", "rel", "via"]].drop_duplicates("dst").itertuples(index=False)][:7]
    nodes = [{"id": "F", "label": firm_id[(code, t)], "ring": 0, "x": 0.0, "y": 0.0, "listed": True, "rel": "", "ev": []}]
    ang = {}
    for i, (d, r) in enumerate(ring1):
        a = 2 * np.pi * i / max(len(ring1), 1) + 0.25
        ang[d] = a
        nodes.append({"id": d, "label": lab(d, r), "ring": 1, "x": float(np.cos(a)), "y": float(np.sin(a)), "listed": d.startswith("C:"),
                      "rel": REL_CN.get(r, r)})
    edges = [{"s": "F", "t": d, "rel": REL_CN.get(r, r)} for d, r in ring1]
    placed = 0
    for d, r, v in ring2:
        vnode = next((x for x, _ in ring1 if x[2:] == str(v) or x == str(v)), None)
        a = ang.get(vnode, 2 * np.pi * (placed + 0.5) / max(len(ring2), 1))
        a = a + (placed % 3 - 1) * 0.22
        placed += 1
        nodes.append({"id": d, "label": lab(d, r), "ring": 2, "x": float(1.9 * np.cos(a)), "y": float(1.9 * np.sin(a)), "listed": d.startswith("C:"),
                      "rel": {"sib_ctrl": "兄弟企业(共同控股股东)", "sib_pe": "其他被投企业(共同创投)", "sc2hop": "客户的客户/供应商的供应商"}[r]})
        edges.append({"s": vnode if vnode else "F", "t": d, "rel": {"sib_ctrl": "共同股东", "sib_pe": "共同创投", "sc2hop": "上下游"}[r]})
    for n in nodes[1:]:
        e = X[X.dst == n["id"]].sort_values("mi")
        n["ev"] = [{"m": int(m), "ym": f"{mi // 12}-{mi % 12 + 1:02d}", "kind": KIND_CN[k]} for m, mi, k in zip(e.m, e.mi, e.kind)]
    # 节点 id 匿名化：内部 id 含股票代码与企业全称，导出前替换为序号
    idmap = {n["id"]: ("F" if n["ring"] == 0 else f"n{i}") for i, n in enumerate(nodes)}
    for n in nodes:
        n["id"] = idmap[n["id"]]
    for e in edges:
        e["s"], e["t"] = idmap.get(e["s"], "F"), idmap.get(e["t"], e["t"])
    # 时间轴事件（观察期内）
    evs = []
    Xin = X[(X.m >= 0) & (X.m <= 11)]
    for (m, dst), g in Xin.groupby(["m", "dst"]):
        r = g.rel.iloc[0]
        evs.append({"m": int(m), "ym": f"{g.mi.iloc[0] // 12}-{g.mi.iloc[0] % 12 + 1:02d}", "lane": LANE.get(r, "股权"),
                    "rel": {"sib_ctrl": "兄弟企业", "sib_pe": "其他被投企业", "sc2hop": "二跳"}.get(r, REL_CN.get(r, r)),
                    "who": labels.get(dst, lab(dst, r)), "kind": "、".join(sorted({KIND_CN[k] for k in g.kind}))})
    lane_cnt = {}
    for e in evs:
        key = (e["m"], e["lane"])
        e["jit"] = (lane_cnt.get(key, 0) % 3 - 1) * 5
        lane_cnt[key] = lane_cnt.get(key, 0) + 1
    months = []
    for r_ in rr.itertuples():
        ix = kmap.get(r_.k)
        c = shp.get(ix)
        g = [[gn, float(sum(c[i] for i, f in enumerate(F4) if f in gs))] for gn, gs in GROUPS] if c is not None else []
        top = []
        if c is not None:
            order = np.argsort(-np.abs(c))[:6]
            for i in order:
                v = DY.at[ix, F4[i]]
                vs = None if pd.isna(v) else (f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}")
                top.append([fname(F4[i]), float(c[i]), vs])
        months.append({"ym": r_.ym, "p1": float(r_.p1), "p4": float(r_.p4), "pd1": float(r_.pd1), "pd4": float(r_.pd4), "ev3": int(r_.ev3),
                       "shap": {"groups": g, "top": top}})
    s0 = D.year * 12 + D.month - 1
    dd = row.def_date
    def_m = None if pd.isna(dd) else int(dd.year * 12 + dd.month - 1 - s0)
    ind = ind_of.get((code, t), "")
    flag_m = [i for i, m in enumerate(months) if m["p4"] >= 0.95]
    flag1 = [i for i, m in enumerate(months) if m["p1"] >= 0.95]
    if row.group in ("def", "miss"):
        focus = int(np.argmax([m["p4"] for m in months[: max(1, min(def_m, 12))]])) if def_m is not None else 11
    else:
        focus = int(np.argmax([m["p4"] for m in months]))
    period = f"{months[0]['ym'].replace('-', '.')}–{months[-1]['ym'].replace('-', '.')}"
    if pd.isna(dd):
        outcome = "观察期内及其后 6 个月未发生首次金融债务违约"
    else:
        outcome = f"{dd.year}-{dd.month:02d} 首次被提起金融债务诉讼" + ("（观察期后）" if def_m is not None and def_m > 11 else "")
    # 叙述
    parts = []
    if flag_m:
        parts.append(f"动态模型在 <b>{months[flag_m[0]]['ym']}</b> 首次把该企业列入预警（分位 {months[flag_m[0]]['p4'] * 100:.1f}）")
        if flag1:
            lead = flag1[0] - flag_m[0]
            parts.append(f"仅用自身特征的模型在 {months[flag1[0]]['ym']} 列入" + (f"，晚 {lead} 个月" if lead > 0 else ("，同月" if lead == 0 else "，更早")))
        else:
            parts.append("仅用自身特征的模型在观察期内始终没有列入预警")
    else:
        parts.append("动态模型在观察期内没有把该企业列入前 5% 预警名单")
    ev_before = [e for e in evs if (def_m is None or e["m"] < def_m)]
    if ev_before:
        e0 = ev_before[0]
        parts.append(f"最早的关联信号是 <b>{e0['who']}</b>（{e0['rel']}）于 {e0['ym']} {e0['kind']}")
    parts.append(f"结局：{outcome}")
    grp_txt = {"def": "被提前预警的违约企业", "miss": "漏检的违约企业：模型没有提前识别，列出来是为了如实展示局限", "fp": "预警后未违约（误报）", "ok": "正常企业"}[row.group]
    story = f"<b>{grp_txt}。</b>" + "；".join(parts) + "。"
    axis = [f"{(s0 + i) // 12}-{(s0 + i) % 12 + 1:02d}" for i in range(12)]
    firms.append({"id": firm_id[(code, t)], "industry": ind, "period": period, "axis": axis, "group": row.group, "outcome": outcome, "months": months,
                  "events": evs, "nodes": nodes, "edges": edges, "def_m": def_m, "def_ym": None if pd.isna(dd) else f"{dd.year}-{dd.month:02d}",
                  "focus_month": focus, "story": story})
order = {"def": 0, "miss": 1, "fp": 2, "ok": 3}
firms.sort(key=lambda f: (order[f["group"]], f["id"]))

# ---------------------------------------------------------------- 预警工作台
R["flag"] = R.p4 >= 0.95
first = R[R.flag].sort_values("S").drop_duplicates(["code", "t"])
first = first.assign(out=((first.def_date > first.S) & (first.def_date <= first.S + pd.DateOffset(months=6))).astype(int))
idx_map = DY.reset_index().set_index(["code", "t", "k"])["index"]
fidx = idx_map.reindex(pd.MultiIndex.from_frame(first[["code", "t", "k"]])).values
shp_all = shap_rows(pd.Index(fidx))
REL_SRC = {"dy_holder_3m": "股东", "dy_codef_3m": "共同被告", "dy_debtor_3m": "债务人", "dy_cus_3m": "客户", "dy_sup_3m": "供应商",
           "dy_ctrl_3m": "同一控制", "dy_sib_ctrl_3m": "兄弟企业", "dy_sib_pe_3m": "其他被投企业", "dy_sc2hop_3m": "二跳主体",
           "dy_affil_3m": "联营/合营", "dy_person_3m": "同一关键人员", "dy_invest_3m": "对外投资", "dy_rp_other_3m": "其他关联方"}
alerts = []
for r_, ix in zip(first.itertuples(), fidx):
    c = shp_all.get(ix)
    drv = max(GROUPS, key=lambda g: sum(c[i] for i, f in enumerate(F4) if f in g[1]))[0] if c is not None else ""
    src = [v for k, v in REL_SRC.items() if k in first.columns and getattr(r_, k, 0) and getattr(r_, k) > 0]
    key = (r_.code, r_.t)
    alerts.append({"ym": r_.ym, "m": int(r_.k), "firm": firm_id.get(key, anon(r_.code)), "industry": ind_of.get(key, ""), "p4": float(r_.p4),
                   "p1": float(r_.p1), "driver": drv, "src": "、".join(src), "src_list": src, "out": int(r_.out), "detail": key in firm_id})
alert_stats = {"n": len(alerts), "hit": float(first.out.mean()), "base": float(R.y6.mean()), "lift": float(first.out.mean() / R.y6.mean())}
al = pd.read_csv(OUT / "dyn_alert.csv")
alert_stats["capture"] = float(al[al.模型 == N_D4].iloc[0]["违约前6个月内被预警比例"])

# ---------------------------------------------------------------- 乘数与证据
mult = pd.read_csv(OUT / "dyn_multipliers.csv")
mults = [{"name": r.iloc[0], "mult": float(r.事件乘数), "lo": float(r.CI_lo), "hi": float(r.CI_hi)} for _, r in mult.iterrows()]
ro = pd.read_csv(OUT / "rolling_oot.csv").set_index("模型")
deltas = []
for n, lab_ in [("M2 +一跳关联", "年度·+一跳关联 vs 自身"), ("M4 +结构位置", "年度·+多跳与结构 vs 自身"), ("S1 无报表+图谱", "年度·无报表+图谱 vs 无报表")]:
    r_ = ro.loc[n]
    deltas.append({"name": lab_, "v": float(r_.ΔAUC), "lo": float(r_.ΔAUC_lo), "hi": float(r_.ΔAUC_hi)})
for (i, j), lab_ in [((2, 0), "月度·+一跳关联时序 vs 自身"), ((3, 0), "月度·+多层级时序 vs 自身"), ((4, 0), "月度·自身+关联时序(无静态图) vs 自身")]:
    fp_ = OUT / f"dyn_delta_{i}_{j}.json"
    if not fp_.exists():
        continue
    d = json.load(open(fp_, encoding="utf-8"))
    deltas.append({"name": lab_, "v": d["d"], "lo": d["lo"], "hi": d["hi"]})
co = pd.read_csv(OUT / "contagion_or.csv")
ors = [{"name": r.关系.replace("(连带)", "").replace("(应收)", ""), "v": float(r.OR), "lo": float(r.OR_lo), "hi": float(r.OR_hi)} for r in co.itertuples()]
mech = pd.read_csv(OUT / "mechanisms.csv")
SHORT = {"毛利率挤压(1SD)": "①毛利率挤压(1SD)", "供应商集中度高": "①供应商集中度高", "挤压×集中度高": "①挤压×集中度高",
         "兄弟企业出险(共同控股股东/同一控制)": "②兄弟企业出险", "其他被投企业出险(共同创投/产业基金)": "②共同创投的其他被投企业出险",
         "同业高速增长者占比(1SD)": "③同业高速增长者占比(1SD)", "本企业增速落后同业": "③本企业增速落后同业", "高速增长者占比×落后": "③高速增长者占比×落后"}
mechs = [{"name": SHORT.get(r.变量, r.变量), "v": float(r["系数(百分点)"]), "lo": float(r.CI_lo), "hi": float(r.CI_hi)} for _, r in mech.iterrows()]
lead = ("数据：A 股非金融上市公司 2016—2023 年逐年关联图谱（10 类关系）与月度关联事件时序；标签：首次以被告身份被提起金融债务诉讼。"
        "所有评估均为时间外推（用过去训练、预测未来）。完整方法与稳健性见<a href=\"./\">分析报告</a>。")
footer = ("原型数据来自中国研究数据服务平台（CNRDS）与国泰安（CSMAR）数据库的衍生结果，企业编号与关联主体均已脱敏，风险分位只用于方法演示，不构成对任何企业的信用评价。"
          "原始数据受许可限制未公开。")
data = {"firms": firms, "alerts": alerts, "alert_stats": alert_stats, "multipliers": mults, "base_rate6": float(R.y6.mean()),
        "evidence": {"lead": lead, "deltas": deltas, "or": ors, "mech": mechs}, "footer": footer}
tpl = (Path(__file__).with_name("app_template.html")).read_text(encoding="utf-8")
html = tpl.replace("{{DATA_JSON}}", json.dumps(data, ensure_ascii=False, separators=(",", ":"))).replace("{{DATE}}", date.today().isoformat())
out = ROOT_DIR / "docs" / "app.html"
out.write_text(html, encoding="utf-8")
print("app:", out, f"{len(html) / 1e6:.2f} MB", "企业", len(firms), "预警", len(alerts), alert_stats)
