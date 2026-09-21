"""贷中监控检验：关联方在决策日之后新出现金融债务/经营性欠款被诉事件，作为触发信号；
比较触发后与未触发时的月度违约率（率比 RR，泊松近似 95% CI），并按样本外自身评分（滚动外推 M1）分层。
剔除本企业自身是当事人（披露方或被列为当事人）的案件，避免把同一宗违约诉讼当成"触发"。"""
import json
import numpy as np
import pandas as pd
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
E = pd.read_parquet(WORK / "edges.parquet")
parties = pd.read_parquet(WORK / "lit_parties.parquet")
cases = pd.read_parquet(WORK / "lit_cases.parquet")
oot = pd.read_parquet(WORK / "oot_rolling.parquet")[["code", "t", "M1 自身"]]

ev = parties[(parties.side == "def") & parties.ctype.isin(["fin", "trade"])].copy()
ev["node"] = np.where(ev.name_code.notna(), "C:" + ev.name_code.astype(str), "E:" + ev.name)
own = cases[cases.self_def & cases.ctype.isin(["fin", "trade"])][["case_id", "code", "date"]].assign(node=lambda d: "C:" + d.code)
EV = pd.concat([ev[["node", "date", "case_id"]], own[["node", "date", "case_id"]]], ignore_index=True).drop_duplicates()
# 企业作为当事人的案件（披露方 或 被列为当事人）
pc = pd.concat([cases[["case_id", "code"]], parties[parties.name_code.notna()][["case_id", "name_code"]].rename(columns={"name_code": "code"})])
pc_set = set(zip(pc.case_id, pc.code))

SETS = {"三条显著渠道(股东/共同被告/债务人)": ["holder", "codef", "debtor"], "全部关系": None,
        "供应链(供应商/客户)": ["sup", "cus"], "三条渠道+1个月反应期": ["holder", "codef", "debtor"],
        "股东+共同被告": ["holder", "codef"], "仅债务人": ["debtor"]}
out_rows, detail = [], []
for name, rels in SETS.items():
    recs = []
    for t in YEARS:
        D = decision_date(t)
        Dend = D + pd.DateOffset(months=12)
        Et = E[E.t == t]
        if rels:
            Et = Et[Et.rel.isin(rels)]
        Et = Et[Et.src.str[2:].isin(P[P.t == t].code)]
        w = EV[(EV.date > D) & (EV.date <= Dend)]
        m = Et[["src", "dst"]].drop_duplicates().merge(w, left_on="dst", right_on="node")
        m["code"] = m.src.str[2:]
        m = m[[(c, k) not in pc_set for c, k in zip(m.case_id, m.code)]]
        trig = m.groupby("code").date.min()
        if "反应期" in name:
            trig = trig + pd.Timedelta(days=30)                      # 银行收到信号后留 1 个月处置时间
        s = P[P.t == t][["code", "y", "event_date"]].copy()
        s["t"] = t
        s["end"] = s.event_date.where(s.y == 1, Dend)
        s["trig"] = s.code.map(trig)
        s.loc[s.trig >= s.end, "trig"] = pd.NaT                     # 违约之后才出现的关联事件不算触发
        s["pre_m"] = ((s.trig.fillna(s.end) - D).dt.days / 30.4375).clip(lower=0)
        s["post_m"] = ((s.end - s.trig).dt.days / 30.4375).fillna(0).clip(lower=0)
        s["post_def"] = ((s.y == 1) & s.trig.notna()).astype(int)
        s["pre_def"] = ((s.y == 1) & s.trig.isna()).astype(int)
        recs.append(s)
    S = pd.concat(recs, ignore_index=True).merge(oot, on=["code", "t"], how="left")
    S["band"] = np.where(S["M1 自身"].isna(), "无样本外评分(2016-18)",
                         np.where(S.groupby("t")["M1 自身"].rank(pct=True) >= 0.8, "自身评分高风险(前20%)", "自身评分中低风险(后80%)"))
    for band, g in [("全部年份", S), *[(b, x) for b, x in S.groupby("band")]]:
        a, pm = g.post_def.sum(), g.post_m.sum()
        b, qm = g.pre_def.sum(), g.pre_m.sum()
        if a == 0 or b == 0:
            rr, lo, hi = np.nan, np.nan, np.nan
        else:
            rr = (a / pm) / (b / qm)
            se = np.sqrt(1 / a + 1 / b)
            lo, hi = rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)
        out_rows.append({"触发口径": name, "分层": band, "被触发企业年": int(g.trig.notna().sum()), "触发后违约": int(a),
                         "触发后月度违约率(‰)": 1000 * a / max(pm, 1e-9), "未触发月度违约率(‰)": 1000 * b / max(qm, 1e-9),
                         "率比RR": rr, "RR_lo": lo, "RR_hi": hi, "触发后违约占全部违约": a / max(a + b, 1)})
    if name == "三条显著渠道(股东/共同被告/债务人)":
        S.to_parquet(WORK / "monitor_detail.parquet", index=False)
        lead = S[(S.post_def == 1)]
        detail = {"触发到违约的月数中位数": float(((lead.end - lead.trig).dt.days / 30.4375).median()),
                  "触发到违约的月数四分位": [float(x) for x in ((lead.end - lead.trig).dt.days / 30.4375).quantile([.25, .75])]}
res = pd.DataFrame(out_rows)
res.to_csv(OUT / "monitor.csv", index=False, encoding="utf-8-sig")
json.dump(detail, open(OUT / "monitor_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res.round(3).to_string(index=False))
print(detail)
