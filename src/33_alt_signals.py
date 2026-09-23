"""补充数据的信号检验：企业自身事件、控股股东层面事件、关联方事件、票据名单、自然灾害与极端天气。

口径与正文一致：月度面板、2019—2023 年滚动外推测试期、标签 y6（未来 6 个月内首次违约）。
每个信号报告：暴露企业-月数、6 个月违约率与未暴露违约率，以及两个条件估计（都回答“模型已知信息之外还有没有增量”）：
① 事件乘数：以样本外自身模型（D1，含自身诉讼、财务与事件时序）的违约对数几率为 offset 的 logit（正文 3.2 节的做法），另控制年份；
② 按「同月 × 自身评分五分位」分层的 MH 优势比。五分位偏粗，出事企业又集中在最高一档，② 会高估，以 ① 为准。
关联关系一律按观察年对齐：企业-月只使用当年图谱里的关联方。
灾害类另行处理：本地灾害的事件乘数控制省份与月份（同一个月里比较受灾省与未受灾省，扣除省份本身的差异），
MH 在「同省同年 × 自身评分五分位」内比较；外省关联方所在地的灾害，乘数控制关联方覆盖省份数与月份，
MH 在「同月 × 关联方覆盖省份数 × 自身评分五分位」内比较（关联方遍布多省的企业更常“暴露”，必须按覆盖省份数比较）。
同一企业同一观察年内的比较也列出，但灾害集中在夏季，这一设计受季节混淆，只备查、不作解释。
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import *

rng = np.random.default_rng(20260923)
ONE_HOP = ["dy_sup_3m", "dy_cus_3m", "dy_holder_3m", "dy_invest_3m", "dy_ctrl_3m", "dy_person_3m",
           "dy_affil_3m", "dy_rp_other_3m", "dy_codef_3m", "dy_debtor_3m", "dy_all_enf_3m"]
D1 = "D1 自身(含自身时序)"
KIND_CN = {"debt_overdue": "债务逾期公告", "bond_default": "债券兑付违约公告", "shixin": "被列为失信被执行人", "acct_frozen": "银行账户被冻结",
           "illegal_guar": "违规担保或担保逾期", "bankruptcy": "被申请破产或重整", "csrc_probe": "被证监会立案调查", "st_new": "新被实施 ST / *ST"}
SH_CN = {"sh_distress": "控股股东或实控人失信、破产重整或债务违约", "share_frozen": "股东所持股份被司法冻结", "pledge_liq": "股东质押平仓或被动减持"}
SHOCK_CN = {"quake": "地震（M≥5）", "typhoon": "台风过境", "rain": "极端降水", "heat": "极端高温", "cold": "寒潮", "disaster": "任一突发灾害（地震、台风、极端降水）"}

dyn = pd.read_parquet(WORK / "dyn_panel.parquet", columns=["code", "t", "k", "S", "y6"] + ONE_HOP)
oot = pd.read_parquet(WORK / "dyn_oot.parquet", columns=["code", "t", "k", D1])
dyn = dyn.merge(oot, on=["code", "t", "k"], how="inner").reset_index(drop=True)
dyn["S"] = pd.to_datetime(dyn.S).astype("datetime64[ns]")
dyn["one"] = (dyn[ONE_HOP].to_numpy() > 0).any(1)
dyn["q5"] = dyn.groupby("S")[D1].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
dyn["ym"] = dyn.S.dt.strftime("%Y-%m")
dyn["st_score"] = dyn.ym + "_" + dyn.q5.astype(str)
master = pd.read_parquet(WORK / "firm_master.parquet")
LOC = pd.read_parquet(WORK / "node_loc.parquet")
own_prov = dict(zip(LOC.node.str[2:], LOC.prov))
dyn["prov"] = dyn.code.map(lambda c: own_prov.get(c))
dyn["st_prov"] = dyn.ym + "_" + dyn.prov.fillna("NA")
KEY = ["code", "t", "k"]
print(f"[面板] {len(dyn):,} 企业-月，基准 6 个月违约率 {dyn.y6.mean():.2%}")


def strat_or(df, expo, strata, reps=250):
    """按 strata 分层的 Mantel-Haenszel 优势比，企业整簇 bootstrap 给 95% 区间。"""
    s = pd.factorize(df[strata])[0]
    x, y, ns = df[expo].to_numpy(bool), df.y6.to_numpy(float), s.max() + 1

    def mh(ix):
        ss, xx, yy = s[ix], x[ix], y[ix]
        a, n1 = np.bincount(ss[xx], yy[xx], ns), np.bincount(ss[xx], minlength=ns)
        c, n0 = np.bincount(ss[~xx], yy[~xx], ns), np.bincount(ss[~xx], minlength=ns)
        n = np.maximum(n1 + n0, 1)
        num, den = (a * (n0 - c) / n).sum(), ((n1 - a) * c / n).sum()
        return num / den if den > 0 else np.nan

    point = mh(np.arange(len(df)))
    _, inv = np.unique(df.code.to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    bs = [mh(np.concatenate([groups[p] for p in rng.integers(0, len(groups), len(groups))])) for _ in range(reps)]
    return float(point), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def multiplier(df, expo, fe=None):
    """事件乘数：以样本外自身模型（D1）的违约对数几率为 offset 的 logit（正文 3.2 节的做法），另加年份固定效应，
    吸收自身模型在不同测试年份的校准偏移（只覆盖部分年份的信号，如票据名单，否则会被高估）；企业聚类标准误。
    fe 给出额外的固定效应列（灾害类用月份 + 省份或关联方覆盖省份数），含月份时不再单独加年份。"""
    p = df[D1].clip(1e-6, 1 - 1e-6).to_numpy()
    fe = fe or []
    parts = [] if "ym" in fe else [pd.get_dummies(df.S.dt.year, prefix="y", drop_first=True, dtype=float)]
    parts += [pd.get_dummies(df[c].fillna("NA").astype(str), prefix=c, drop_first=True, dtype=float) for c in fe]
    X = pd.concat(parts, axis=1)
    X.insert(0, "x", df[expo].to_numpy(float))
    b, se = logit_offset(df.y6.to_numpy(float), sm.add_constant(X.to_numpy()), np.log(p / (1 - p)), pd.factorize(df.code)[0])
    return float(np.exp(b[1])), float(np.exp(b[1] - 1.96 * se[1])), float(np.exp(b[1] + 1.96 * se[1]))


def logit_offset(y, X, off, groups, tol=1e-10, iters=50):
    """带 offset 的 logit（IRLS 解正规方程），企业聚类稳健标准误，小样本修正与 statsmodels 相同。
    statsmodels 的 GLM 用 lstsq 求解，设计矩阵有 90 多列固定效应时要申请很大的工作区，这里直接解 k×k 的正规方程。"""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + off)))
        w = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < tol:
            break
    mu = 1 / (1 + np.exp(-(X @ beta + off)))
    A_inv = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    G = groups.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, groups, X * (y - mu)[:, None])
    n, k = X.shape
    V = A_inv @ (S.T @ S) @ A_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(V))


rows = []


FE_CN = {None: "年份", ("prov", "ym"): "省份、月份", ("deg_b", "ym"): "关联方覆盖省份数、月份"}


def report(group, name, expo, strata="st_score", df=None, fe=None):
    d = dyn if df is None else df
    n = int(d[expo].sum())
    if n < 30:
        print(f"  {name}: 暴露 {n}，跳过")
        return
    k = int(d.loc[d[expo], "y6"].sum())
    r1, r0 = k / n, d.loc[~d[expo], "y6"].mean()
    o, lo, hi = strat_or(d, expo, strata)
    m, mlo, mhi = multiplier(d, expo, fe) if k else (np.nan, np.nan, np.nan)
    rows.append({"类别": group, "信号": name, "暴露企业-月": n, "6个月违约": k, "违约率": r1, "未暴露违约率": r0, "倍数": r1 / r0 if r0 else np.nan,
                 "分层优势比": o, "OR_lo": lo, "OR_hi": hi, "分层口径": STRATA_CN[strata], "事件乘数": m, "乘数_lo": mlo, "乘数_hi": mhi,
                 "乘数另控制": FE_CN[tuple(fe) if fe else None]})
    print(f"  {name:<28} 暴露 {n:>7,}  违约率 {r1:6.2%} vs {r0:6.2%}  OR {o:5.2f} [{lo:4.2f}, {hi:5.2f}]  "
          f"乘数 {m:5.2f} [{mlo:4.2f}, {mhi:5.2f}]  {STRATA_CN[strata]}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "alt_signals.csv", index=False, encoding="utf-8-sig")   # 每条落盘，防中途崩溃丢结果


STRATA_CN = {"st_score": "同月×自身评分五分位", "st_prov": "同月×本企业所在省", "st_provyear": "同省同年×自身评分五分位",
             "st_deg": "同月×关联方覆盖省份数×自身评分五分位"}


def flag_from_events(ev_code_date, window_months, label, base=None):
    """ev_code_date: 已映射到“被暴露企业”的 code/date 表 → 企业-月是否在过去 window 个月内出现过事件。"""
    on = ["code", "t"] if "t" in ev_code_date.columns else ["code"]           # 关联关系按观察年成立，带 t 时只作用于当年的月份
    e = ev_code_date[on + ["date"]].drop_duplicates()
    m = dyn[KEY + ["S"]].merge(e, on=on, how="inner")
    m = m[(m.date <= m.S) & (m.date > m.S - pd.DateOffset(months=window_months))]
    hit = m[KEY].drop_duplicates().assign(**{label: True})
    dyn.drop(columns=[label], errors="ignore", inplace=True)
    merged = dyn[KEY].merge(hit, on=KEY, how="left")[label]
    dyn[label] = merged.fillna(False).astype(bool).values


# ---------------------------------------------------------------- 1. 企业自身事件
EV = pd.read_parquet(WORK / "alt_events.parquet")
EV["date"] = pd.to_datetime(EV.date).astype("datetime64[ns]")
dfd = pd.read_parquet(WORK / "dyn_panel.parquet", columns=["code", "def_date"]).dropna().drop_duplicates()
dfd["def_date"] = pd.to_datetime(dfd.def_date).astype("datetime64[ns]")
lead = {}
for group, kinds in [("企业自身（巨潮公告、ST）", KIND_CN), ("控股股东层面（上市公司公告）", SH_CN)]:
    print(f"\n[{group}]")
    for kind, cn in kinds.items():
        flag_from_events(EV[EV.kind == kind], 12, "x")
        report(group, cn, "x")
        m = dfd.merge(EV[EV.kind == kind][["code", "date"]], on="code")      # 违约前 24 个月内首次出现该事件，到违约的月数
        m = m[(m.date < m.def_date) & (m.date >= m.def_date - pd.DateOffset(months=24))]
        first = m.groupby(["code", "def_date"]).date.min().reset_index()
        if len(first):
            lead[cn] = {"违约中有此前兆": int(len(first)), "提前月数中位数": float(((first.def_date - first.date).dt.days / 30.4375).median())}
for r in rows:
    if r["信号"] in lead:
        r.update(lead[r["信号"]])

P = pd.read_parquet(WORK / "pledge_panel.parquet")
P["date"] = pd.to_datetime(P.date).astype("datetime64[ns]")
dyn["S"] = pd.to_datetime(dyn.S).astype("datetime64[ns]")
P = P.sort_values("date")
pm = pd.merge_asof(dyn[KEY + ["S"]].sort_values("S"), P[["code", "date", "pledge_ratio"]].rename(columns={"date": "S"}).sort_values("S"),
                   on="S", by="code", direction="backward", tolerance=pd.Timedelta(days=100))
pm = pm.set_index(KEY).reindex(pd.MultiIndex.from_frame(dyn[KEY]))
dyn["pledge_ratio"] = pm.pledge_ratio.values
cover = dyn.pledge_ratio.notna().mean()
for thr in [30, 50]:
    dyn["x"] = dyn.pledge_ratio.fillna(0) >= thr
    report("控股股东层面（股权质押）", f"股权质押比例 ≥ {thr}%", "x")
print(f"  质押数据覆盖 {cover:.1%} 的企业-月，最晚 {P.date.max():%Y-%m}")

# ---------------------------------------------------------------- 2. 上市关联方的事件（沿一跳关系传给本企业）
E = pd.read_parquet(WORK / "edges.parquet")
E = E[E.src.str.startswith("C:") & E.dst.str.startswith("C:") & (E.src != E.dst)][["src", "dst", "t"]].drop_duplicates()
E["code"], E["ncode"] = E.src.str[2:], E.dst.str[2:]
print(f"\n[上市关联方] 上市公司之间的一跳关系 {len(E):,} 条（按年）")


def neighbor_events(ev):
    m = E.merge(ev.rename(columns={"code": "ncode"}), on="ncode")
    return m[["code", "t", "date"]]


for kind, cn in KIND_CN.items():
    flag_from_events(neighbor_events(EV[EV.kind == kind][["code", "date"]]), 12, "x")
    report("上市关联方事件", "关联方：" + cn, "x")
all_ev = neighbor_events(EV[EV.kind.isin(list(KIND_CN))][["code", "date"]])
flag_from_events(all_ev, 12, "x")
report("上市关联方事件", "关联方：任一上述事件", "x")
report("上市关联方事件", "关联方：任一上述事件（仅一跳无诉讼信号的月份）", "x", df=dyn[~dyn.one])

# 票交所承兑人名单（26 号抓取，27 号首检）：关联关系按观察年对齐后重算，并补事件乘数，与本表其他信号同一口径
Bl = pd.read_parquet(WORK / "bill_events.parquet")
Bl["date"] = pd.to_datetime(Bl.date).astype("datetime64[ns]")
_full = master.dropna(subset=["full"])
name2code = dict(zip(_full.full.map(norm_name), _full.code))
Bl["node"] = np.where(Bl.name.map(name2code).notna(), "C:" + Bl.name.map(name2code).astype(str), "E:" + Bl.name)
Eb = pd.read_parquet(WORK / "edges.parquet", columns=["src", "dst", "t"])
Eb = Eb[Eb.src.str.startswith("C:") & (Eb.src != Eb.dst)].drop_duplicates()
lk = Eb.merge(Bl[["node", "date", "kind"]], left_on="dst", right_on="node")
lk["code"] = lk.src.str[2:]
print(f"\n[票据名单] 关联方命中 {lk.node.nunique():,} 个主体")
for kinds, lab in [(["overdue"], "关联方上过承兑人逾期或持续逾期名单"), (["undisclosed"], "关联方上过信用信息未披露名单")]:
    flag_from_events(lk[lk.kind.isin(kinds)][["code", "t", "date"]], 12, "x")
    report("票据承兑人名单（票交所）", lab, "x")
flag_from_events(lk[["code", "t", "date"]], 12, "x")
report("票据承兑人名单（票交所）", "关联方上过任一名单（仅一跳无诉讼信号的月份）", "x", df=dyn[~dyn.one])
own_bill = Bl[Bl.node.str.startswith("C:")].assign(code=lambda d: d.node.str[2:])
flag_from_events(own_bill[["code", "date"]], 12, "x")
report("票据承兑人名单（票交所）", "企业自己上过任一名单", "x")

# 发债主体（含不上市的集团母公司、控股股东）的违约、展期与破产重整公告，沿本企业的全部一跳关系传入
BE = WORK / "bond_events.parquet"
if BE.exists():
    B = pd.read_parquet(BE)
    B["date"] = pd.to_datetime(B.date).astype("datetime64[ns]")
    Ea = pd.read_parquet(WORK / "edges.parquet", columns=["src", "dst", "t"])
    Ea = Ea[Ea.src.str.startswith("C:") & (Ea.src != Ea.dst)].drop_duplicates()
    m = Ea.merge(B[["node", "date"]].drop_duplicates(), left_on="dst", right_on="node")
    m["code"] = m.src.str[2:]
    for lab, sub in [("关联发债主体违约、展期或破产重整（含非上市）", m), ("其中：非上市发债主体", m[m.node.str.startswith("E:")])]:
        flag_from_events(sub[["code", "t", "date"]], 12, "x")
        report("关联发债主体（巨潮债券公告）", lab, "x")

# ---------------------------------------------------------------- 3. 非上市关联方的工商登记状态
REGP = WORK / "node_registry.parquet"
if REGP.exists():
    R = pd.read_parquet(REGP)
    dead = R[R.dead & R.approve.notna()][["node", "approve"]].rename(columns={"approve": "date"})
    if len(dead):
        Ea = pd.read_parquet(WORK / "edges.parquet")
        Ea = Ea[Ea.src.str.startswith("C:") & Ea.dst.isin(set(dead.node))][["src", "dst", "t"]].drop_duplicates()
        m = Ea.merge(dead, left_on="dst", right_on="node")
        m["code"] = m.src.str[2:]
        flag_from_events(m[["code", "t", "date"]], 12, "x")
        report("非上市关联方（工商登记）", "关联方注销、吊销或停业", "x")
    else:
        print("\n[工商] 匹配到的关联方里没有注销或吊销记录（该工商库只含存续企业），跳过此项")

# ---------------------------------------------------------------- 4. 自然灾害与极端天气
S = pd.read_parquet(WORK / "prov_shocks.parquet")
shock_months = {}
for c in SHOCK_CN:
    s = S[S[c]][["prov", "ym"]]
    shock_months[c] = set(zip(s.prov, s.ym))


def lagged(ym, n=3):
    p = pd.Period(ym, "M")
    return [(p - i).strftime("%Y-%m") for i in range(n)]


lags = {ym: lagged(ym) for ym in dyn.ym.unique()}
dyn["st_provyear"] = dyn.prov.fillna("NA") + "_" + dyn.S.dt.year.astype(str) + "_" + dyn.q5.astype(str)
print("\n[灾害·本地] 本企业所在省近 3 个月（同省同年内比较，排除省份构成差异）")
for c, cn in SHOCK_CN.items():
    hit = shock_months[c]
    dyn["x"] = [any((p, y) in hit for y in lags[ym]) if isinstance(p, str) else False for p, ym in zip(dyn.prov, dyn.ym)]
    report("本地灾害与天气", "本地：" + cn, "x", strata="st_provyear", fe=["prov", "ym"])

Eall = pd.read_parquet(WORK / "edges.parquet", columns=["src", "dst", "t"])
Eall = Eall[Eall.src.str.startswith("C:")].drop_duplicates()
Eall["code"] = Eall.src.str[2:]
Eall = Eall.merge(LOC.rename(columns={"node": "dst", "prov": "nprov"})[["dst", "nprov"]], on="dst", how="left").dropna(subset=["nprov"])
Eall["oprov"] = Eall.code.map(lambda c: own_prov.get(c))
Eall = Eall[Eall.nprov != Eall.oprov]                                            # 只看外省关联方，排除本地灾害
nprovs = Eall.groupby(["code", "t"]).nprov.agg(lambda x: tuple(sorted(set(x))))
del Eall
dyn["nprovs"] = [nprovs.get((c, t), ()) for c, t in zip(dyn.code, dyn.t)]
npv = dyn.nprovs.map(len)
dyn["deg_b"] = pd.cut(npv, [-1, 0, 1, 2, 4, 7, 99], labels=["0", "1", "2", "3-4", "5-7", "8+"]).astype(str)
dyn["st_deg"] = dyn.ym + "_" + dyn.deg_b + "_" + dyn.q5.astype(str)             # 同月、关联方覆盖省份数相近、自身评分相近
dyn["st_firmyear"] = dyn.code + "_" + dyn.t.astype(str)
STRATA_CN["st_firmyear"] = "同一企业同一观察年内"
sub = dyn[npv > 0].copy()
print(f"\n[灾害·外省关联方] 至少有一个外省关联方的企业-月 {len(sub):,}；关联方所在省近 3 个月（不含本企业所在省）")
for c, cn in SHOCK_CN.items():
    hit = shock_months[c]
    sub["x"] = [any((p, y) in hit for p in ps for y in lags[ym]) for ps, ym in zip(sub.nprovs, sub.ym)]
    report("外省关联方灾害与天气", "关联方所在地：" + cn, "x", strata="st_deg", df=sub, fe=["deg_b", "ym"])
    report("外省关联方灾害与天气", "关联方所在地：" + cn, "x", strata="st_firmyear", df=sub, fe=["deg_b", "ym"])   # 企业-年内比较受季节混淆，仅备查

out = pd.DataFrame(rows)
out.to_csv(OUT / "alt_signals.csv", index=False, encoding="utf-8-sig")
meta = {"企业月": int(len(dyn)), "基准": float(dyn.y6.mean()), "质押覆盖": float(cover), "质押最晚": f"{P.date.max():%Y-%m}",
        "定位来源": LOC.src.value_counts(dropna=False).to_dict(), "上市-上市一跳关系": int(len(E))}
json.dump(meta, open(OUT / "alt_signals_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(f"\n[done] {len(out)} 条信号 → {OUT / 'alt_signals.csv'}")
