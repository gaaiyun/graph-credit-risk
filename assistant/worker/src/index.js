/**
 * 关联风险全景 · AI 风控助手（Cloudflare Worker + Workers AI）
 *
 * 检索在浏览器完成：页面把命中的企业风险画像与方法知识库段落作为“资料”随问题发来，
 * Worker 只负责校验来源、限流、拼装系统提示词并以流式返回模型回答。
 */
const ORIGINS = ["https://gaaiyun.github.io", "http://localhost:8807", "http://127.0.0.1:8807", "http://localhost:8808", "http://127.0.0.1:8808"];
const MODEL = "@cf/zai-org/glm-4.7-flash";

const SYSTEM = `你是“关联风险全景”的 AI 风控助手，服务对象是银行对公信贷、贷后管理与风险管理人员。

先在内部判断问题属于下面哪一类，再按该类要求作答；回答中不要写出类别编号或类别名称：
A. 概念、方法与公式类（例如“什么是出险”“事件乘数怎么算”“多跳传导公式”“预警怎么分级”）：直接给出定义或计算方法；有公式时写出公式，并逐一解释符号的含义；需要时补充本研究得到的关键数值。
B. 企业风险类（问题里出现公司名称或股票代码）：概括这家企业在图谱中的风险画像，依次说明风险状态与违约记录、样本外违约概率、公告风险事件、股权质押、票据名单，以及有风险事件的关联方和关系类型，最后指出最值得关注的传导渠道。
C. 报告与产品类（研究结论、数据来源、产品功能与用法）：按资料直接回答。

规则：
1. 只依据用户消息里【资料】部分的内容作答，数字、日期与事件一律照资料引用，不补充资料以外的信息。资料里没有问题所问的企业时，说明图谱数据中没有该企业，并提示输入全称、简称或股票代码。
2. 用简体中文，专业、肯定、简洁。开头一句话直接回答问题，不写“结论”“依据”之类的标题，不写客套话；内容多时用短列表，关键数字可用 **加粗**。
3. 方法资料中的总体统计（各类事件乘数、命中率、率比等）描述的是全样本规律，不是某家企业自己的数值。谈具体企业时，企业的数字只取自“企业风险画像”；如需引用总体统计，要写明“全样本中这类事件的乘数为 X”。
4. 公式用纯文本书写，例如 logit P(y=1) = logit(p_self) + α_年份 + β·x，不要使用 LaTeX、$ 符号或代码块。
5. 违约概率与风险状态是模型对公开数据的测度结果，照实陈述数值与口径，不给投资建议。`;

const json = (body, status, headers) =>
  new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json; charset=utf-8" } });

export default {
  async fetch(req, env) {
    const origin = req.headers.get("Origin") || "";
    const allowed = ORIGINS.includes(origin);
    const cors = {
      "Access-Control-Allow-Origin": allowed ? origin : ORIGINS[0],
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400",
      Vary: "Origin",
    };
    if (req.method === "OPTIONS") return new Response(null, { headers: cors });
    const url = new URL(req.url);
    if (url.pathname === "/health") return json({ ok: true, model: MODEL }, 200, cors);
    if (req.method !== "POST" || url.pathname !== "/chat") return json({ error: "not found" }, 404, cors);
    if (!allowed) return json({ error: "forbidden" }, 403, cors);

    const ip = req.headers.get("CF-Connecting-IP") || "anon";
    if (env.LIMITER) {
      const { success } = await env.LIMITER.limit({ key: ip });
      if (!success) return json({ error: "提问过于频繁，请一分钟后再试。" }, 429, cors);
    }
    let body;
    try { body = await req.json(); } catch { return json({ error: "bad request" }, 400, cors); }
    const question = String(body.question || "").trim().slice(0, 500);
    if (!question) return json({ error: "empty question" }, 400, cors);
    const context = String(body.context || "").slice(0, 14000);
    const history = (Array.isArray(body.history) ? body.history : []).slice(-6)
      .map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: String(m.content || "").slice(0, 1500) }));
    const messages = [
      { role: "system", content: SYSTEM },
      ...history,
      { role: "user", content: `【资料】\n${context || "（没有检索到相关资料）"}\n\n【问题】${question}` },
    ];
    try {
      const stream = await env.AI.run(MODEL, { messages, stream: true, max_tokens: 1400, temperature: 0.3, chat_template_kwargs: { enable_thinking: false } });
      return new Response(stream, { headers: { ...cors, "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache" } });
    } catch (e) {
      return json({ error: "模型服务暂时不可用，请稍后再试。", detail: String(e).slice(0, 200) }, 502, cors);
    }
  },
};
