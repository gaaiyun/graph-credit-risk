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

FEAT_CN = dict(FEAT_CN)
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


# ---------------------------------------------------------------- 图3 两种评分口径的增量（滚动外推 2019-2023）
ro = pd.read_csv(OUT / "rolling_oot.csv").set_index("模型")
gr = pd.read_csv(OUT / "gnn_rolling.csv")
rows = []
for n, lab in [("M2 +一跳关联", "+一跳关联"), ("M3 +多跳传导", "+多跳传导"), ("M4 +结构位置", "+结构位置"), ("M5 +同业竞争", "+同业竞争"),
               ("S1 无报表+图谱", "无报表 → 无报表+图谱")]:
    r = ro.loc[n]
    rows.append(("年度评分", lab, r.ΔAUC, r.ΔAUC_lo, r.ΔAUC_hi))
g = gr[(gr.模型 == "R-GCN") & (gr.对照 == "MLP(同架构去边)")].iloc[0]
rows.append(("年度评分", "R-GCN vs 同架构去边网络", g.ΔAUC, g.CI_lo, g.CI_hi))
for (i, j), lab in [((1, 0), "+静态图谱"), ((3, 1), "+一跳与多层级关联时序"), ((3, 0), "自身 → 全部图谱信息"), ((4, 0), "+关联时序(不含静态图谱)")]:
    d = json.load(open(OUT / f"dyn_delta_{i}_{j}.json", encoding="utf-8"))
    rows.append(("月度动态评分", lab, d["d"], d["lo"], d["hi"]))
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
yy, ylabels, yv = [], [], 0
for grp in ["年度评分", "月度动态评分"]:
    sub = [r for r in rows if r[0] == grp]
    ax.text(-0.034, yv + 0.1, grp, fontsize=8, color=INK, fontweight="bold", va="bottom")
    yv -= 0.45
    for _, lab, d, lo, hi in sub:
        c = ACC if lo > 0 else (ACC2 if hi < 0 else MUTED)
        ax.hlines(yv, lo, hi, color=c, lw=1.8)
        ax.scatter([d], [yv], color=c, s=16, zorder=3)
        ax.text(hi + 0.0012, yv, f"{d:+.4f}", va="center", fontsize=6.8, color=INK)
        yy.append(yv); ylabels.append(lab)
        yv -= 1
    yv -= 0.3
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_yticks(yy)
ax.set_yticklabels(ylabels, fontsize=7.2)
ax.set_xlim(-0.035, 0.03)
ax.set_xlabel("ΔAUC 与 95% 置信区间（年度：配对分层 bootstrap；月度：按企业整簇 bootstrap）", fontsize=7.2, color=MUTED)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / "fig3_ablation.png")

# ---------------------------------------------------------------- 图4 左：年度传导渠道 OR；右：月度动态事件乘数
co = pd.read_csv(OUT / "contagion_or.csv").iloc[::-1].reset_index(drop=True)
mu = pd.read_csv(OUT / "dyn_multipliers.csv").iloc[::-1].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), dpi=200)


def forest_ax(ax, labels, v, lo, hi, p=None, xt=(0.5, 1, 2, 3), xl=(0.3, 4.5)):
    y = np.arange(len(v))
    col = [ACC if l_ > 1 else (ACC2 if h_ < 1 else MUTED) for l_, h_ in zip(lo, hi)]
    ax.hlines(y, np.clip(lo, xl[0], xl[1]), np.clip(hi, xl[0], xl[1]), color=col, lw=1.7)
    ax.scatter(v, y, color=col, s=15, zorder=3)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(*xl)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(t) for t in xt])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.9)
    for yy_, vv, hh in zip(y, v, hi):
        ax.text(min(hh, xl[1]) * 1.06, yy_, f"{vv:.2f}", va="center", fontsize=6.5, color=INK)
    clean(ax)


forest_ax(axes[0], [r.replace("(连带)", "(担保连带)") for r in co.关系], co.OR.values, co.OR_lo.values, co.OR_hi.values)
axes[0].set_title("年度截面：存在出险关联方的优势比", fontsize=8, color=INK, loc="left")
axes[0].set_xlabel("控制自身财务、年份与行业；公司聚类", fontsize=6.8, color=MUTED)
forest_ax(axes[1], mu["关联事件(近3个月)"].values, mu.事件乘数.values, mu.CI_lo.values, mu.CI_hi.values, xt=(0.5, 1, 2, 4, 8), xl=(0.3, 9))
axes[1].set_title("月度动态：近 3 个月关联事件的乘数", fontsize=8, color=INK, loc="left")
axes[1].set_xlabel("未来 6 个月违约几率的倍数，以样本外自身模型评分为基准", fontsize=6.8, color=MUTED)
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
