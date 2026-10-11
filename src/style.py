"""报告图表与网页共用的视觉规范：深蓝、钢蓝、青绿、金、珊瑚五色为主，按数值用渐变，不显著与对照用灰阶。"""
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
DEEP, STEEL, JADE, GOLD, CORAL = "#16324F", "#2F6DB5", "#2A9D8F", "#E9A23B", "#E4572E"
SOFT = "#CDD3DC"


def ramp(colors, v, lo=None, hi=None):
    """按数值在给定色带上取色，用于单系列柱的渐变着色。"""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    v = np.asarray(v, float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    return LinearSegmentedColormap.from_list("r", colors)((v - lo) / max(hi - lo, 1e-9))


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
