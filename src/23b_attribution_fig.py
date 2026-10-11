"""图 fig9_attribution：预警名单的风险来源构成（左）与按触发渠道的命中率（右），读 23 号脚本保存的结果。"""
import json

import numpy as np
import pandas as pd

from common import *
from style import *

apply()
comp = pd.read_csv(OUT / "attribution_groups.csv", index_col=0)
chan = pd.read_csv(OUT / "attribution_channels.csv")
att = json.load(open(OUT / "attribution.json", encoding="utf-8"))
n_all, n_hit = att["预警数"], att["命中数"]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=200, gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
cols = {"自身财务": "#A9B4C4", "自身事件与治理": "#D5DCE6", "静态图谱结构": BRAND_2, "一跳关联事件时序": BRAND, "多层级时序": NAVY}
cats = list(comp.columns)
left = np.zeros(len(cats))
for g in comp.index:
    v = comp.loc[g].values
    ax.barh(range(len(cats))[::-1], v * 100, left=left * 100, color=cols[g], height=0.55, label=g, edgecolor="white", lw=0.8)
    for i, (l_, w_) in enumerate(zip(left, v)):
        if w_ > 0.07:
            ax.text((l_ + w_ / 2) * 100, len(cats) - 1 - i, f"{w_ * 100:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if g in ("一跳关联事件时序", "多层级时序", "静态图谱结构") else INK)
    left = left + v
ax.set_yticks(range(len(cats))[::-1])
ax.set_yticklabels([f"{c}（{n}）" for c, n in zip(cats, [n_all, n_hit, n_all - n_hit])], fontsize=7)
ax.set_xlim(0, 100)
ax.set_xlabel("正向 SHAP 贡献的平均构成（%）", fontsize=7)
ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
ax.set_title("预警名单的风险来源构成", fontsize=8, loc="left")
clean(ax)

ax = axes[1]
ch = chan.iloc[::-1]
y = np.arange(len(ch))
base = n_hit / n_all
cc = [COMPARE if c == "无关联触发" else BRAND for c in ch.渠道]
ax.barh(y, ch["6个月违约率"] * 100, color=cc, height=0.6, zorder=2)
ax.hlines(y, ch.CI_lo * 100, ch.CI_hi * 100, color=INK2, lw=0.8, zorder=3)
for yy, r in zip(y, ch.itertuples()):
    ax.text(r.CI_hi * 100 + 0.6, yy, f"{r[3] * 100:.1f}%（n={r[2]}）", va="center", fontsize=6.3, color=INK)
ax.axvline(base * 100, color=MUTED, lw=0.8, ls="--")
ax.text(base * 100, len(ch) - 0.4, f" 全部预警 {base:.1%}", fontsize=6.5, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels(ch.渠道, fontsize=7)
ax.set_xlabel("预警后 6 个月内违约率（%，横线为 Wilson 95% 区间）", fontsize=7)
ax.set_title("按触发渠道：预警的命中率", fontsize=8, loc="left")
ax.set_xlim(0, max(ch.CI_hi) * 100 * 1.35)
clean(ax, "x")
fig.tight_layout()
fig.savefig(OUT / "fig" / "fig9_attribution.png", bbox_inches="tight", pad_inches=0.05)
print("fig9 saved")
