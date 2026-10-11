"""报告图表与网页共用的视觉规范：品牌蓝为主色，风险红只用于违约与出险，其余用灰阶。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#0B1F3B"      # 深海军蓝：标题级强调
BRAND = "#1D5BD8"     # 品牌蓝：主数据系列、显著结果
BRAND_2 = "#7FA3E8"   # 浅品牌蓝：次级系列
INK = "#1D2433"       # 正文
INK2 = "#4A5568"      # 次级文字
MUTED = "#8A94A6"     # 轴、注释、不显著结果
GRID = "#E3E8EF"      # 网格与对照填充
COMPARE = "#C5CFDD"   # 对照组柱
RISK = "#D92D20"      # 违约
WARN = "#F79009"      # 出险 / 关注
TEAL, ORANGE, VIOLET = "#0E9F8E", "#F28C28", "#6E56CF"   # 分类色（已过色盲与对比度校验）
CATS = [BRAND, ORANGE, TEAL, VIOLET, "#C93C6E"]


def apply():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean(ax, grid_axis=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
