"""报告附录 B 的产品界面截图：首页、全量关联图谱、产品原型四个模块、AI 风控助手问答。

依赖 docs/ 下已发布的网站（12、22、24、35 号产物）。用本机 Chrome 无头模式截图，AI 助手的两张会实际调用线上 Worker 作答。
输出 output/fig/shots/b*.jpg（12 号放进附录 B）与正文图 8 的 fig10_graph_product.png（在 34 号之后运行）。
"""
import functools
import http.server
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from common import *

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS = ROOT_DIR / "docs"
SH = OUT / "fig" / "shots"
SH.mkdir(parents=True, exist_ok=True)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


# 端口 8807 在 Worker 的来源白名单内，AI 助手才能作答
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8807), functools.partial(Quiet, directory=str(DOCS)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:8807"
HIDE_RA = "document.querySelectorAll('.ra-launch,.ra-panel').forEach(e => e.remove()); document.documentElement.classList.remove('ra-docked')"

# (文件名, 路径, 等待秒数, 截图前执行的脚本, 是否保留 AI 助手)
SHOTS = [
    ("b1_home", "/index.html", 3, None, True),
    ("b2_graph", "/graph/#y=2023&n=600221", 8, None, False),
    ("b3_graph_risk", "/graph/#y=2023", 7, "document.getElementById('riskOnly').click()", False),
    ("b4_pano", "/app.html?v=pano", 3, None, False),
    ("b5_alert", "/app.html?v=alert", 3, None, False),
    ("b6_sim", "/app.html?v=sim", 3, None, False),
    ("b7_evid", "/app.html?v=evid", 4, None, False),
    ("b8_ai_firm", "/graph/#y=2023&n=600221", 8, "RiskAssistant.open('海航控股的风险情况怎么样？')", True),
    ("b9_ai_method", "/index.html", 3, "RiskAssistant.open('事件乘数怎么计算？')", True),
]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
    for name, path, wait, js, keep_ra in SHOTS:
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        pg.goto(BASE + path, wait_until="networkidle")
        time.sleep(wait)
        if js:
            pg.evaluate(js)
            time.sleep(2)
        if "RiskAssistant.open('" in (js or ""):
            # 等流式回答写完：参考来源出现即结束
            pg.wait_for_selector(".ra-src", timeout=90000)
            time.sleep(1)
        if not keep_ra:
            pg.evaluate(HIDE_RA)
            time.sleep(1)
        png = SH / f"{name}.png"
        pg.screenshot(path=str(png))
        Image.open(png).convert("RGB").save(SH / f"{name}.jpg", quality=84)
        png.unlink()
        pg.close()
        print("[截图]", name)
    b.close()
srv.shutdown()

# 正文图 8：图谱浏览器（去掉左侧筛选栏）+ 客户全景视图（去掉侧栏），覆盖 34 号的同名输出
g = Image.open(SH / "b2_graph.jpg")
g = g.crop((544, 0, g.width, g.height))
a = Image.open(SH / "b4_pano.jpg")
a = a.crop((448, 0, a.width, a.height))
H = 1100
g = g.resize((int(g.width * H / g.height), H), Image.LANCZOS)
a = a.resize((int(a.width * H / a.height), H), Image.LANCZOS)
gap = 28
canvas = Image.new("RGB", (g.width + gap + a.width + 4, H + 4), "white")
for im, x in ((g, 0), (a, g.width + gap)):
    canvas.paste(Image.new("RGB", (im.width + 4, H + 4), "#D2D2D7"), (x, 0))
    canvas.paste(im, (x + 2, 2))
canvas.save(OUT / "fig" / "fig10_graph_product.png")
print("[图 8]", canvas.size)
