"""描述性分析：样本与标签结构、图规模、各类关系覆盖率、有风险邻居时的违约率（单变量传导证据）、行业分布。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)
INK, MUTED, ACC, ACC2, GRID = "#1f2933", "#7b8794", "#c2410c", "#2563eb", "#e4e7eb"

P = pd.read_parquet(WORK / "panel.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
REL_LAB = {"sup": "供应商", "cus": "客户", "holder": "股东", "invest": "对外投资", "ctrl": "同一控制", "person": "同一关键人员",
           "affil": "联营/合营/子公司", "rp_other": "其他关联方", "codef": "共同被告", "debtor": "债务人"}

# 1. 年度样本
yr = P.groupby("t").agg(样本=("y", "size"), 违约=("y", "sum"), 违约率=("y", "mean")).reset_index()
g = E.groupby("t").agg(边=("src", "size"))
g["边"] //= 2
nodes = E.groupby("t").apply(lambda d: len(set(d.src) | set(d.dst)))
yr = yr.merge(g, left_on="t", right_index=True).assign(节点=yr.t.map(nodes))
yr["决策日"] = yr.t.map(lambda t: str(decision_date(t).date()))
yr.to_csv(OUT / "desc_year.csv", index=False, encoding="utf-8-sig")
print(yr.to_string(index=False))

# 2. 关系类型：边数（2023 观察年）、覆盖率、有风险邻居占比、条件违约率与提升倍数
base = P.y.mean()
rows = []
for r, lab in REL_LAB.items():
    has = P[f"nb_{r}_n"] > 0
    risky = P[f"nb_{r}_risky"] > 0
    rows.append({"关系": lab, "key": r, "边数(2023)": int((E[(E.t == 2023) & (E.rel == r)].shape[0])),
                 "覆盖率": has.mean(), "有风险邻居占比": risky.mean(),
                 "有风险邻居违约率": P.y[risky].mean(), "无风险邻居违约率": P.y[~risky].mean(),
                 "提升倍数": P.y[risky].mean() / P.y[~risky].mean()})
rel = pd.DataFrame(rows)
rel.to_csv(OUT / "desc_relations.csv", index=False, encoding="utf-8-sig")
print(rel.round(4).to_string(index=False))

# 3. 行业（申万一级，最新名称映射）
P["行业"] = P.sw_l1.map(SW_L1).fillna("其他")
ind = P.groupby("行业").agg(样本=("y", "size"), 违约=("y", "sum"), 违约率=("y", "mean")).sort_values("违约", ascending=False)
ind.to_csv(OUT / "desc_industry.csv", encoding="utf-8-sig")
print(ind.head(12).to_string())

# 4. 违约组 vs 正常组关键变量对比
cmp_cols = {"lev": "资产负债率", "roa": "ROA(%)", "cfo_to_debt": "现金债务总额比", "cash_ratio": "现金比率", "impair_to_rev": "减值/营收(%)",
            "own_trade": "经营性欠款被诉(衰减)", "own_viol": "违规处罚(衰减)", "hop1_all": "一跳关联风险", "ppr3": "三跳扩散风险",
            "group_fin": "集团圈内违约暴露", "peer_fin_rate": "同业违约率"}
cmp = P.groupby("y")[list(cmp_cols)].median().T.rename(index=cmp_cols, columns={0: "正常组中位数", 1: "违约组中位数"})
cmp["违约组均值"] = P[P.y == 1][list(cmp_cols)].mean().values
cmp["正常组均值"] = P[P.y == 0][list(cmp_cols)].mean().values
cmp.to_csv(OUT / "desc_compare.csv", encoding="utf-8-sig")
print(cmp.round(4).to_string())

# 图 1：年度违约率 + 样本数
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
ax.bar(yr.t, yr.样本, color=GRID, width=0.62, label="样本企业数")
ax.set_ylabel("样本企业数", color=MUTED, fontsize=8)
ax.tick_params(axis="both", labelsize=8, colors=MUTED)
ax2 = ax.twinx()
ax2.plot(yr.t, yr.违约率 * 100, color=ACC, marker="o", lw=1.8, ms=4, label="首次违约率")
for x_, y_ in zip(yr.t, yr.违约率 * 100):
    ax2.annotate(f"{y_:.2f}%", (x_, y_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=ACC)
ax2.set_ylabel("首次违约率 (%)", color=ACC, fontsize=8)
ax2.tick_params(axis="y", labelsize=8, colors=ACC)
ax2.set_ylim(0, yr.违约率.max() * 100 * 1.35)
for a in (ax, ax2):
    for s in ["top"]:
        a.spines[s].set_visible(False)
ax.set_xticks(yr.t)
ax.set_xticklabels([f"{t}\n({t+1}.4)" for t in yr.t], fontsize=7)
ax.set_xlabel("观察年 t（括号内为决策时点）", fontsize=8, color=MUTED)
ax.axvspan(2020.5, 2021.5, color="#fde68a", alpha=0.35, lw=0)
ax.axvspan(2021.5, 2023.5, color="#bfdbfe", alpha=0.35, lw=0)
ax.text(2018, ax.get_ylim()[1] * 0.93, "训练", ha="center", fontsize=8, color=MUTED)
ax.text(2021, ax.get_ylim()[1] * 0.93, "验证", ha="center", fontsize=8, color=MUTED)
ax.text(2022.5, ax.get_ylim()[1] * 0.93, "测试(外推)", ha="center", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(FIG / "fig1_year.png")

# 图 2：有/无风险邻居的违约率（按关系类型）
rel_s = rel.sort_values("提升倍数")
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
yy = np.arange(len(rel_s))
ax.barh(yy + 0.19, rel_s.有风险邻居违约率 * 100, height=0.36, color=ACC, label="存在有风险事件的该类邻居")
ax.barh(yy - 0.19, rel_s.无风险邻居违约率 * 100, height=0.36, color="#cbd2d9", label="不存在")
for i, r_ in enumerate(rel_s.itertuples()):
    ax.text(r_.有风险邻居违约率 * 100 + 0.1, i + 0.19, f"×{r_.提升倍数:.1f}", va="center", fontsize=7, color=INK)
ax.set_yticks(yy)
ax.set_yticklabels([f"{a}（覆盖{b:.0%}）" for a, b in zip(rel_s.关系, rel_s.有风险邻居占比)], fontsize=7.5)
ax.set_xlabel("未来12个月首次违约率 (%)", fontsize=8, color=MUTED)
ax.tick_params(axis="x", labelsize=8, colors=MUTED)
ax.axvline(base * 100, color=MUTED, lw=0.8, ls="--")
ax.text(base * 100, len(rel_s) - 0.3, f" 全样本 {base:.2%}", fontsize=7, color=MUTED)
ax.legend(fontsize=7, frameon=False, loc="lower right")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig2_relation_lift.png")
print("figs saved")
