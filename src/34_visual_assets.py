"""网站与报告的视觉素材：全量关联网络渲染图、产品截图、报告图 10（网络 + 产品界面）。

依赖 22 号（docs/app.html）与 24 号 --publish（docs/graph/）的产物。截图用本机 Chrome 无头模式，
脚本内起一个本地静态服务读 docs/ 目录。产出：
- docs/assets/hero_graph.jpg        2023 年全量关联网络（白底宽幅，出险为琥珀色、违约为红色）
- docs/assets/shot_*.jpg            产品原型与图谱浏览器截图
- output/fig/fig10_graph_product.png 报告用：左为网络渲染，右为产品界面
"""
import functools
import http.server
import json
import threading
import time
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

from common import *
from style import *

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS = ROOT_DIR / "docs"
ASSETS = DOCS / "assets"
ASSETS.mkdir(exist_ok=True)
GX = DOCS / "graph" / "data"
BG = "#070D1C"


def load_js(path, key):
    t = path.read_text(encoding="utf-8")
    tag = f'GXD["{key}"]='
    return json.loads(t[t.index(tag) + len(tag):].rstrip().rstrip(";"))


# ---------------------------------------------------------------- 1. 全量网络渲染
nodes = load_js(GX / "nodes.js", "nodes")
yr = load_js(GX / "y2023.js", "y2023")
n = nodes["n"]
xy = np.asarray(nodes["xy"], float).reshape(-1, 2)
types = np.frombuffer(nodes["types"].encode(), dtype=np.uint8) - ord("0")
P = np.asarray(yr["pairs"], int).reshape(-1, 4)
deg = np.bincount(np.r_[P[:, 0], P[:, 1]], minlength=n)
risk = np.asarray(yr["risk"], int).reshape(-1, 5)
fin = np.zeros(n, int)
fin[risk[:, 0]] = risk[:, 1]
firms = np.asarray(yr["firms"], int).reshape(-1, 6)
ydef = np.full(n, -1)
ydef[firms[:, 0]] = firms[:, 1]
present = (deg > 0) | (ydef >= 0)
st = np.where(ydef == 1, 2, np.where(fin > 0, 1, 0))
lo, hi = np.quantile(xy[present], 0.004, axis=0), np.quantile(xy[present], 0.996, axis=0)

def render(path, W, H, px):
    """白底渲染：上市公司为品牌蓝，其他主体为浅蓝灰，出险为琥珀色，违约为红色。"""
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    seg = np.stack([xy[P[:, 0]], xy[P[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#0071E3", linewidths=0.22, alpha=0.07, zorder=1))
    size = np.where(types == 0, np.minimum(1.6 + 0.5 * np.sqrt(deg), 9), np.minimum(0.8 + 0.35 * np.sqrt(deg), 7))
    base = present & (st == 0)
    ax.scatter(*xy[base & (types != 0)].T, s=size[base & (types != 0)] ** 2 * 0.35, c="#A3B8D8", alpha=0.75, lw=0, zorder=2)
    ax.scatter(*xy[base & (types == 0)].T, s=size[base & (types == 0)] ** 2 * 0.45, c="#3B7DDD", alpha=0.85, lw=0, zorder=3)
    for k, col in ((1, "#F5A623"), (2, "#E5352B")):
        m = present & (st == k)
        ax.scatter(*xy[m].T, s=size[m] ** 2 * 2.4, c=col, alpha=0.12, lw=0, zorder=4)
        ax.scatter(*xy[m].T, s=size[m] ** 2 * 0.6, c=col, alpha=0.95, lw=0, zorder=5)
    pad = (hi - lo) * 0.02
    cx, cy, half = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi - lo).max() / 2 + pad.max()
    ax.set_xlim(cx - half * W / H, cx + half * W / H)
    ax.set_ylim(cy + half, cy - half)                           # 与浏览器的 y 轴方向一致
    ax.set_aspect("equal")
    ax.axis("off")
    tmp = path.with_suffix(".png")
    fig.savefig(tmp, facecolor="white")
    plt.close(fig)
    Image.open(tmp).convert("RGB").resize(px, Image.LANCZOS).save(path, quality=88)
    tmp.unlink()


render(ASSETS / "hero_graph.jpg", 16, 8, (2400, 1200))
render(OUT / "fig" / "graph_square.jpg", 10, 10, (1600, 1600))
print(f"[网络] 节点 {int(present.sum()):,}，关系 {len(P):,}，出险 {int((present & (st == 1)).sum()):,}，违约 {int((st == 2).sum())}")

# ---------------------------------------------------------------- 2. 产品截图
from playwright.sync_api import sync_playwright


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(DOCS)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{srv.server_address[1]}"
SHOTS = [("shot_pano", "/app.html?v=pano", 1440, 900, 2), ("shot_alert", "/app.html?v=alert", 1440, 900, 2),
         ("shot_graph", "/graph/?v=shot#y=2023&n=002713", 1440, 860, 7)]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
    for name, path, w, h, wait in SHOTS:
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.goto(base_url + path, wait_until="networkidle")
        time.sleep(wait)
        pg.evaluate("document.querySelectorAll('.ra-launch,.ra-panel').forEach(e => e.remove())")
        pg.screenshot(path=str(ASSETS / f"{name}.png"))
        Image.open(ASSETS / f"{name}.png").convert("RGB").save(ASSETS / f"{name}.jpg", quality=86)
        (ASSETS / f"{name}.png").unlink()
        pg.close()
    b.close()
srv.shutdown()
print("[截图]", ", ".join(s[0] for s in SHOTS))

# ---------------------------------------------------------------- 3. 报告图 10
g = Image.open(OUT / "fig" / "graph_square.jpg")
a = Image.open(ASSETS / "shot_pano.jpg")
H = 1000
g = g.resize((H, H), Image.LANCZOS)
a = a.resize((int(a.width * H / a.height), H), Image.LANCZOS)
gap = 24
canvas = Image.new("RGB", (g.width + gap + a.width + 2, H + 2), "#D2D2D7")
canvas.paste(Image.new("RGB", (g.width + gap, H + 2), "white"), (0, 0))
canvas.paste(g, (0, 1))
canvas.paste(a, (g.width + gap + 1, 1))
canvas.save(OUT / "fig" / "fig10_graph_product.png")
print("[图10]", canvas.size)
