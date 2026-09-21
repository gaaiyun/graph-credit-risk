"""结果图：消融增量（固定切分 + 滚动外推）、传导强度森林图、SHAP、累计捕获曲线。"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)
INK, MUTED, ACC, ACC2, GRID, POS = "#1f2933", "#7b8794", "#c2410c", "#2563eb", "#e4e7eb", "#15803d"

FEAT_CN = {
    "impair_to_rev": "资产减值/营收", "sw_l1_c": "申万一级行业", "fund_n": "持股基金家数", "cfo_to_due": "现金流到期债务保障",
    "firm_age": "上市年限", "cash_ratio": "现金比率", "tang_lev": "有形资产负债率", "top1_stake": "第一大股东持股",
    "log_ta": "总资产(对数)", "fin_exp_ratio": "财务费用率", "wc_to_loan": "营运资金/借款", "d_roa": "ROA变动",
    "lev": "资产负债率", "d_cfo_to_debt": "现金债务比变动", "inst_ratio": "机构持股比例", "cfo_to_debt": "现金债务总额比",
    "int_debt_ratio": "带息负债比率", "quick_ratio": "速动比率", "cur_ratio": "流动比率", "rel_rev_growth": "相对同业营收增速",
    "pagerank": "PageRank中心度", "period_exp_ratio": "期间费用率", "nb_codef_n": "共同被告数", "own_viol": "违规处罚(衰减)",
    "ltloan_to_ta": "长期借款/总资产", "roa": "ROA", "roe": "ROE", "gross_margin": "毛利率", "net_margin": "净利率",
    "own_trade": "经营性欠款被诉", "own_other_def": "其他案件被诉", "own_def_n12": "近12月被诉次数", "own_def_amt12": "近12月被诉金额",
    "hop1_all": "一跳关联风险", "hop2_all": "二跳关联风险", "ppr3": "三跳扩散风险", "deg": "关联度数", "deg_listed": "上市关联方数",
    "kcore": "k-core", "group_size": "控制圈规模", "group_fin": "控制圈违约暴露", "group_risk": "控制圈风险", "peer_fin_rate": "同业违约率",
    "peer_roa": "同业ROA", "rel_roa": "相对同业ROA", "peer_rev_growth": "同业营收增速", "rel_d_roa": "相对同业ROA变动",
    "log_rev": "营业收入(对数)", "rev_growth": "营收增速", "eq_mult": "权益乘数", "int_cover": "利息保障倍数", "cfo_to_cl": "现金流动负债比",
    "cfo_to_intdebt": "现金带息债务比", "ebit_margin": "息税前利润率", "cfo_to_profit": "现金/利润总额", "d_lev": "负债率变动",
    "d_gross_margin": "毛利率变动", "d_cur_ratio": "流动比率变动", "loss": "当年亏损", "loss_2y": "连续两年亏损", "own_probe12": "近12月立案调查",
    "sell_major": "大股东净减持", "sell_exec": "高管净减持", "sell_person": "个人股东净减持", "top1_person": "第一大股东为自然人",
    "soe": "国有控股", "top10_nonfin": "前十大非金融股东持股", "board_c": "上市板块", "own_fin_old": "24-36月前违约",
    "own_sec_fraud": "虚假陈述被诉", "own_viol_severe": "严重违规处罚", "peer_n": "同业家数", "peer_loss_rate": "同业亏损率",
    "peer_d_roa": "同业ROA变动", "group_listed": "控制圈上市公司数", "grs": "图谱风险分GRS",
}
REL_CN = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员",
          "affil": "联营/合营/子公司", "rp_other": "其他关联方", "codef": "共同被告", "debtor": "债务人", "creditor": "债权人"}


def feat_cn(c):
    if c in FEAT_CN:
        return FEAT_CN[c]
    for pre, lab in [("nb_", "一跳"), ("hop2_", "二跳")]:
        if c.startswith(pre):
            rest = c[len(pre):]
            for k, v in REL_CN.items():
                if rest.startswith(k + "_") or rest == k:
                    suf = rest[len(k):].strip("_")
                    sm = {"n": "数量", "fin": "金融违约", "trade": "欠款被诉", "risky": "风险邻居数", "dist": "财务困境", "": ""}
                    return f"{lab}·{v}{sm.get(suf, suf)}"
    return c


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED)


# ---------------------------------------------------------------- 图3 消融增量
ab = pd.read_csv(OUT / "ablation.csv")
ro = pd.read_csv(OUT / "rolling_oot.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200, gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
names = ab.模型.tolist()
x = np.arange(len(names))
ax.errorbar(x, ab.AUC, yerr=[ab.AUC - ab.AUC_lo, ab.AUC_hi - ab.AUC], fmt="o", color=INK, ms=4, capsize=2.5, lw=1)
for i, r in ab.iterrows():
    ax.text(i, r.AUC_hi + 0.002, f"{r.AUC:.3f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
ax.set_ylabel("测试集 AUC（2022-2023）", fontsize=8, color=MUTED)
ax.set_title("固定切分：各模型 AUC 与 95% CI", fontsize=8.5, color=INK, loc="left")
clean(ax)
ax = axes[1]
rr = ro[ro.对照.notna()].copy()
rr = rr[rr.模型.str.startswith(("M", "S1", "R1"))]
y = np.arange(len(rr))[::-1]
col = [POS if lo > 0 else (ACC if hi < 0 else MUTED) for lo, hi in zip(rr.ΔAUC_lo, rr.ΔAUC_hi)]
ax.hlines(y, rr.ΔAUC_lo, rr.ΔAUC_hi, color=col, lw=1.6)
ax.scatter(rr.ΔAUC, y, color=col, s=16, zorder=3)
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([f"{m}  vs {c.split(' ')[0]}" for m, c in zip(rr.模型, rr.对照)], fontsize=7)
for yy, r in zip(y, rr.itertuples()):
    ax.text(r.ΔAUC_hi + 0.001, yy, f"{r.ΔAUC:+.4f}", va="center", fontsize=6.8, color=INK)
ax.set_xlabel("ΔAUC（滚动外推 2019-2023 合并，配对bootstrap 95% CI）", fontsize=7.5, color=MUTED)
ax.set_title("滚动外推：相对基线的增量", fontsize=8.5, color=INK, loc="left")
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "fig3_ablation.png")

# ---------------------------------------------------------------- 图4 传导强度森林图
co = pd.read_csv(OUT / "contagion_or.csv")
co = co.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
y = np.arange(len(co))
col = [POS if r.OR_lo > 1 else (ACC2 if r.OR_hi < 1 else MUTED) for r in co.itertuples()]
col = [ACC if c == POS else c for c in col]
ax.hlines(y, co.OR_lo, co.OR_hi, color=col, lw=1.8)
ax.scatter(co.OR, y, color=col, s=18, zorder=3)
ax.axvline(1, color=MUTED, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 3])
ax.set_xticklabels(["0.5", "1", "2", "3"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(y)
lab = [f"{r.关系}" + (f"（覆盖{r.覆盖率:.1%}）" if pd.notna(r.覆盖率) else "") for r in co.itertuples()]
ax.set_yticklabels(lab, fontsize=7.5)
for yy, r in zip(y, co.itertuples()):
    star = "***" if r.p < 0.01 else ("**" if r.p < 0.05 else ("*" if r.p < 0.1 else ""))
    ax.text(max(r.OR_hi, 1) * 1.05, yy, f"{r.OR:.2f}{star}", va="center", fontsize=7, color=INK)
ax.set_xlabel("优势比 OR（控制自身财务、年份与行业固定效应；公司聚类稳健标准误）", fontsize=7.5, color=MUTED)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "fig4_contagion.png")

# ---------------------------------------------------------------- 图5 SHAP
imp = pd.read_csv(OUT / "shap_importance.csv", index_col=0).iloc[:, 0]
meta = json.load(open(OUT / "model_meta.json", encoding="utf-8"))
gs = pd.Series(meta["group_share"]).reindex(["自身", "一跳关联", "多跳传导", "结构位置", "同业竞争"]).fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, gridspec_kw={"width_ratios": [0.8, 1.2]})
ax = axes[0]
cols = [MUTED, ACC, ACC, ACC, ACC2]
ax.barh(np.arange(len(gs))[::-1], gs.values * 100, color=cols, height=0.6)
for i, v in enumerate(gs.values):
    ax.text(v * 100 + 1, len(gs) - 1 - i, f"{v:.1%}", va="center", fontsize=7, color=INK)
ax.set_yticks(np.arange(len(gs))[::-1])
ax.set_yticklabels(gs.index, fontsize=7.5)
ax.set_xlabel("平均|SHAP|占比（%）", fontsize=7.5, color=MUTED)
ax.set_title("特征组贡献", fontsize=8.5, color=INK, loc="left")
clean(ax)
ax = axes[1]
top = imp.head(18)[::-1]
graph_like = lambda c: c.startswith(("nb_", "hop", "ppr", "group_", "deg", "pagerank", "kcore"))
comp_like = lambda c: c.startswith(("peer_", "rel_"))
cc = [ACC if graph_like(c) else (ACC2 if comp_like(c) else "#9aa5b1") for c in top.index]
ax.barh(np.arange(len(top)), top.values, color=cc, height=0.65)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([feat_cn(c) for c in top.index], fontsize=7)
ax.set_xlabel("平均|SHAP|（测试集，M5 全特征模型）", fontsize=7.5, color=MUTED)
ax.set_title("前 18 个特征（橙=图谱，蓝=同业，灰=自身）", fontsize=8.5, color=INK, loc="left")
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "fig5_shap.png")

# ---------------------------------------------------------------- 图6 贷中关联事件预警：触发后累计违约 + 分层率比
S = pd.read_parquet(WORK / "monitor_detail.parquet")
mon = pd.read_csv(OUT / "monitor.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200, gridspec_kw={"width_ratios": [1, 1.05]})
ax = axes[0]
grid = np.arange(0, 12.01, 0.25)
trg = S[S.trig.notna()].copy()
trg["dur"] = (trg.end - trg.trig).dt.days / 30.4375
un = S.copy()
un["dur"] = un.pre_m                                     # 未触发期间：从决策日到触发/违约/窗口结束
un["ev"] = un.pre_def


def km(dur, ev):
    dur, ev = np.asarray(dur), np.asarray(ev)
    surv, out = 1.0, []
    for g0, g1 in zip(np.r_[0, grid[:-1]], grid):
        at = (dur >= g0).sum()
        d = ((dur >= g0) & (dur < g1) & (ev == 1)).sum()
        if at > 0:
            surv *= 1 - d / at
        out.append(1 - surv)
    return np.array(out)


ax.plot(grid, km(trg.dur, trg.post_def) * 100, color=ACC, lw=1.8, label=f"关联方新出现债务诉讼后（n={len(trg)}）")
ax.plot(grid, km(un.dur, un.ev) * 100, color=MUTED, lw=1.6, ls="--", label=f"未触发期间（n={len(un)}）")
ax.set_xlabel("自触发（或决策日）起的月数", fontsize=7.5, color=MUTED)
ax.set_ylabel("累计首次违约率（%）", fontsize=7.5, color=MUTED)
ax.legend(fontsize=6.8, frameon=False, loc="upper left")
ax.set_title("触发后累计违约（Kaplan-Meier）", fontsize=8.5, color=INK, loc="left")
clean(ax)
ax = axes[1]
show = [("三条显著渠道(股东/共同被告/债务人)", "全部年份", "三条渠道·全部"),
        ("三条显著渠道(股东/共同被告/债务人)", "自身评分中低风险(后80%)", "三条渠道·自身评分中低风险"),
        ("三条显著渠道(股东/共同被告/债务人)", "自身评分高风险(前20%)", "三条渠道·自身评分高风险"),
        ("三条渠道+1个月反应期", "全部年份", "三条渠道·留1个月反应期"),
        ("股东+共同被告", "全部年份", "仅股东与共同被告"),
        ("仅债务人", "全部年份", "仅债务人"),
        ("供应链(供应商/客户)", "全部年份", "供应链(供应商/客户)"),
        ("全部关系", "全部年份", "全部关系")]
rows = [mon[(mon.触发口径 == a) & (mon.分层 == b)].iloc[0] for a, b, _ in show]
y = np.arange(len(rows))[::-1]
for yy, r, (_, _, lab) in zip(y, rows, show):
    c = ACC if r.RR_lo > 1 else MUTED
    ax.hlines(yy, r.RR_lo, r.RR_hi, color=c, lw=1.8)
    ax.scatter([r.率比RR], [yy], color=c, s=18, zorder=3)
    ax.text(r.RR_hi * 1.06, yy, f"{r.率比RR:.2f}", va="center", fontsize=7, color=INK)
ax.axvline(1, color=MUTED, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks([1, 2, 4, 8])
ax.set_xticklabels(["1", "2", "4", "8"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(y)
ax.set_yticklabels([s_[2] for s_ in show], fontsize=7)
ax.set_xlabel("触发后/未触发 月度违约率之比（95% CI）", fontsize=7.5, color=MUTED)
ax.set_title("率比：按自身评分分层与口径", fontsize=8.5, color=INK, loc="left")
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "fig6_monitor.png")
print("figures done")
