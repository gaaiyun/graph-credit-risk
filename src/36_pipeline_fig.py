"""报告图：风险模型全流程（黑白线条）。数据 → 图谱与时序 → 模型 → 业务应用，底部为验证口径。
输出 output/fig/fig0_pipeline.png，由 12 号放在 3.1 节。
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from common import *
from style import apply

apply()
INK, GRAY, LIGHT = "#111111", "#555555", "#F2F2F2"
W, H = 170, 74
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, y0, x1, y1, title, lines=(), fill="white", dash=False, tsize=7.4, lsize=6.2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6", lw=0.7, ec=INK, fc=fill,
                                ls=(0, (3, 2)) if dash else "-", zorder=2))
    n = len(lines)
    top = (y0 + y1) / 2 + (n * 2.75) / 2
    ax.text((x0 + x1) / 2, top, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK, zorder=3)
    for k, ln in enumerate(lines):
        ax.text((x0 + x1) / 2, top - 3.1 - k * 2.75, ln, ha="center", va="center", fontsize=lsize, color=GRAY, zorder=3)


def arrow(x0, y0, x1, y1, dash=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=INK, shrinkA=0, shrinkB=0.4,
                                 ls=(0, (3, 2)) if dash else "-", zorder=1))


def line(xs, ys, dash=False):
    ax.plot(xs, ys, color=INK, lw=0.7, ls=(0, (3, 2)) if dash else "-", zorder=1, solid_capstyle="butt")


# 列标题
COLS = [(1, 35, "数据"), (43, 81, "图谱与时序"), (90, 125, "模型"), (134, 169, "业务应用")]
for k, (x0, x1, t) in enumerate(COLS):
    cx = (x0 + x1) / 2
    ax.add_patch(plt.Circle((cx - 9.5, 69.2), 2.1, fc=INK, ec=INK, zorder=3))
    ax.text(cx - 9.5, 69.1, str(k + 1), ha="center", va="center", fontsize=6.4, color="white", fontweight="bold", zorder=4)
    ax.text(cx - 6, 69.2, t, ha="left", va="center", fontsize=8, fontweight="bold", color=INK)
    line([x0, x1], [65.6, 65.6])

# ① 数据
box(1, 13, 35, 62, "公开披露数据", ["诉讼与执行 · CNRDS", "关联交易与股权 · CNRDS", "前五大客户与供应商 · CSMAR", "票据逾期名单 · 票交所",
                                  "风险公告 · 巨潮资讯", "股权质押 · 中国结算", "工商登记", "", "每条记录带公告日"], fill=LIGHT)
# ② 图谱与时序
box(43, 39, 81, 62, "时点一致的逐年异质图谱", ["决策日：次年 4 月 30 日", "只用当日已披露的信息", "10 类关系 · 6.8 万主体"])
box(43, 13, 81, 36, "衰减事件时序", ["w = 0.5^(m / 12)", "近 1、3、12 个月事件累计", "一跳与多层级关系"])
line([35, 39], [37.5, 37.5]); line([39, 39], [24.5, 50.5])
arrow(39, 50.5, 43, 50.5); arrow(39, 24.5, 43, 24.5)
# ③ 模型
box(90, 48, 125, 62, "年度评分卡", ["自身财务、诉讼与治理", "LightGBM"])
box(90, 30.5, 125, 44.5, "月度动态模型", ["223 个特征", "回声消除的多跳传导"])
box(90, 13, 125, 27, "事件乘数与 SHAP 归因", ["logit P = logit(p_self) + α + β·x", "乘数 = exp(β)"])
line([81, 85.5], [50.5, 50.5]); line([81, 85.5], [24.5, 24.5]); line([85.5, 85.5], [20, 55])
for y in (55, 37.5, 20):
    arrow(85.5, y, 90, y)
# ④ 业务应用
APP = [(54.5, 62, "准入、定价与额度", "评分卡只用自身信息"), (44.5, 52, "预警名单", "月度风险分前 5%"),
       (34.5, 42, "分级复核", "优先复核 / 常规复核"), (24.5, 32, "风险归因", "回溯到关联主体与路径"),
       (12.5, 20.5, "AI 风控助手", "检索企业画像与方法文档作答")]
for y0, y1, t, l in APP:
    box(134, y0, 169, y1, t, [l], tsize=7, lsize=5.9)
arrow(125, 58.2, 134, 58.2)                       # 评分卡 → 准入
arrow(125, 37.5, 129.5, 37.5); line([129.5, 129.5], [37.5, 48.2]); arrow(129.5, 48.2, 134, 48.2)   # 动态模型 → 预警名单
arrow(151.5, 44.5, 151.5, 42)                     # 名单 → 分级
line([125, 131.5], [20, 20]); line([131.5, 131.5], [20, 38.2]); arrow(131.5, 38.2, 134, 38.2); arrow(131.5, 28.2, 134, 28.2)  # 乘数 → 分级、归因
arrow(125, 16.2, 134, 16.2, dash=True)            # 画像与方法 → 助手

# 验证口径
box(43, 2, 169, 9, "验证：2019—2023 年逐年滚动样本外检验 · 同一企业内比较（MH 优势比）· 行业×年份固定效应",
    dash=True, tsize=6.4)
fig.savefig(OUT / "fig" / "fig0_pipeline.png", dpi=300, facecolor="white")
print("[图] fig0_pipeline.png")
