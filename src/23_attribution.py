"""风险归因分析（服务于"贷中预警 + 风险归因"主线）：
对测试期（2019-2023 滚动外推）每个企业-观察年首次进入月度前 5% 预警名单的月份：
A. 用对应折的 D4 模型计算 SHAP，按 自身财务 / 自身事件与治理 / 静态图谱结构 / 一跳关联事件时序 / 多层级时序 归组，
   统计正向贡献的来源构成、主要来源分布，并比较 6 个月内违约与未违约的预警。
B. 触发渠道：预警当月近 3 个月内有新增风险事件的关联关系类型，比较各渠道预警的 6 个月违约率。
C. 违约企业的可归因比例：违约前 6 个月内至少有一次关联方新增风险事件的比例。"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import importlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from statsmodels.stats.proportion import proportion_confint


M19 = importlib.import_module("19_dynamic_model")
M4 = importlib.import_module("04_model")
DY, SETS = M19.DY, M19.SETS
names = list(SETS)
N_D1, N_D4 = names[0], names[3]
F4 = SETS[N_D4]
G_OWN_FIN = set(M4.FIN) | {c for c in F4 if c.startswith(("q_", "dq_"))}
GROUPS = [("自身财务", G_OWN_FIN), ("自身事件与治理", (set(M4.OWN_EV) | set(M19.OWN_DYN)) - G_OWN_FIN),
          ("静态图谱结构", set(M4.HOP1 + M4.MULTI + M4.STRUCT)), ("一跳关联事件时序", set(M19.DYN1)), ("多层级时序", set(M19.DYNM))]
REL_G = ["静态图谱结构", "一跳关联事件时序", "多层级时序"]
gidx = {g: np.array([i for i, f in enumerate(F4) if f in s]) for g, s in GROUPS}

R = pd.read_parquet(WORK / "dyn_oot.parquet")
R["p4"] = R.groupby("S")[N_D4].rank(pct=True)
first = R[R.p4 >= 0.95].sort_values("S").drop_duplicates(["code", "t"]).copy()
first["out"] = ((first.def_date > first.S) & (first.def_date <= first.S + pd.DateOffset(months=6))).astype(int)
key = DY.reset_index().set_index(["code", "t", "k"])["index"]
first["ix"] = key.reindex(pd.MultiIndex.from_frame(first[["code", "t", "k"]])).values

# ---------------- A. SHAP 来源构成
contrib = np.zeros((len(first), len(F4)))
for t in sorted(first.t.unique()):
    b = lgb.Booster(model_file=str(WORK / f"dyn_D4_fold{t}.txt"))
    m = (first.t == t).values
    X = DY.loc[first.ix.values[m], F4].to_numpy(np.float32)
    contrib[m] = b.predict(X, pred_contrib=True)[:, :-1]
G = pd.DataFrame({g: contrib[:, ix].sum(1) for g, ix in gidx.items()}, index=first.index)
Gpos = G.clip(lower=0)
share = Gpos.div(Gpos.sum(1).replace(0, np.nan), axis=0)
first["rel_share"] = share[REL_G].sum(1)
first["primary"] = Gpos.idxmax(axis=1)
first["primary_kind"] = np.where(first.primary.isin(REL_G), "关联方", "自身")
comp = pd.DataFrame({"全部预警": share.mean(), "6 个月内违约": share[first.out == 1].mean(), "未违约": share[first.out == 0].mean()})
prim = pd.crosstab(first.primary, first.out, normalize="columns")
print("预警数", len(first), "命中", int(first.out.sum()))
print(comp.round(3))
print(prim.round(3))

# ---------------- B. 触发渠道
REL = {"dy_cus_3m": "客户", "dy_holder_3m": "股东", "dy_codef_3m": "共同被告(担保连带)", "dy_debtor_3m": "债务人", "dy_sup_3m": "供应商",
       "dy_ctrl_3m": "同一控制", "dy_sib_ctrl_3m": "兄弟企业(共同控股股东)", "dy_sib_pe_3m": "其他被投企业(共同创投)", "dy_sc2hop_3m": "二跳上下游",
       "dy_affil_3m": "联营/合营/子公司", "dy_person_3m": "同一关键人员", "dy_invest_3m": "对外投资", "dy_rp_other_3m": "其他关联方"}
trig = DY.loc[first.ix.values, list(REL)].fillna(0).gt(0).set_index(first.index)
enf = DY.loc[first.ix.values, "dy_all_enf_3m"].fillna(0).gt(0).values
first["any_trig"] = trig.any(axis=1).values
chan = []
for c, lab in REL.items():
    m = trig[c].values
    if m.sum() >= 10:
        chan.append({"渠道": lab, "预警数": int(m.sum()), "6个月违约率": float(first.out.values[m].mean())})
chan.append({"渠道": "关联方被强制执行/失信", "预警数": int(enf.sum()), "6个月违约率": float(first.out.values[enf].mean())})
chan.append({"渠道": "无关联触发", "预警数": int((~first.any_trig).sum()), "6个月违约率": float(first.out.values[~first.any_trig.values].mean())})
chan = pd.DataFrame(chan)
lo_, hi_ = proportion_confint((chan.预警数 * chan["6个月违约率"]).round().astype(int), chan.预警数, method="wilson")
chan["CI_lo"], chan["CI_hi"] = lo_, hi_
chan = chan.sort_values("6个月违约率", ascending=False)
print(chan.round(3).to_string(index=False))

# ---------------- C. 违约企业的可归因比例（违约前 6 个月内有关联方新增事件）
evc = [c for c in DY.columns if c.startswith("dy_") and c.endswith("_3m") and not c.startswith(("dy_all_", "dy_peer_"))]
Rm = R.merge(DY[["code", "t", "k"] + evc], on=["code", "t", "k"], how="left")
pre = Rm[Rm.def_date.notna() & (Rm.def_date > Rm.S) & (Rm.def_date <= Rm.S + pd.DateOffset(months=6))]
has = pre.assign(tr=pre[evc].fillna(0).sum(1) > 0).groupby(["code", "def_date"]).tr.any()
nondef = Rm[Rm.def_date.isna()]
base_tr = (nondef[evc].fillna(0).sum(1) > 0).mean()
res = {"预警数": int(len(first)), "命中数": int(first.out.sum()), "命中率": float(first.out.mean()),
       "预警中主要来源为关联方的比例": float((first.primary_kind == "关联方").mean()),
       "命中预警中主要来源为关联方的比例": float((first[first.out == 1].primary_kind == "关联方").mean()),
       "预警中有关联触发的比例": float(first.any_trig.mean()),
       "有关联触发的预警命中率": float(first.out[first.any_trig].mean()), "无关联触发的预警命中率": float(first.out[~first.any_trig].mean()),
       "关联正向贡献占比_全部": float(first.rel_share.mean()), "关联正向贡献占比_命中": float(first.rel_share[first.out == 1].mean()),
       "关联正向贡献占比_未命中": float(first.rel_share[first.out == 0].mean()),
       "违约事件数": int(len(has)), "违约前6个月有关联方新增事件的比例": float(has.mean()),
       "未违约企业-月有关联方新增事件的比例": float(base_tr)}
# 预警分级：按触发渠道（由上面的渠道命中率给出分级依据）
L1 = trig["dy_cus_3m"].values | trig["dy_codef_3m"].values | enf
L2 = ~L1 & (trig["dy_holder_3m"].values | trig["dy_rp_other_3m"].values | trig["dy_sc2hop_3m"].values | trig["dy_debtor_3m"].values)
first["level"] = np.where(L1, 1, np.where(L2, 2, 3))
lv = first.groupby("level").out.agg(["size", "sum", "mean"])
lo_l, hi_l = proportion_confint(lv["sum"], lv["size"], method="wilson")
res["分级"] = {int(k): {"n": int(r["size"]), "hit": int(r["sum"]), "rate": float(r["mean"]), "lo": float(a), "hi": float(b)}
             for (k, r), a, b in zip(lv.iterrows(), lo_l, hi_l)}
res["一级占违约命中比例"] = float(first[first.level == 1].out.sum() / max(first.out.sum(), 1))
# 规则固定不变，按测试期前后两段检查稳定性（渠道选择来自全测试期，存在事后选择风险，这里只看是否两段都成立）
res["分段"] = {}
for lab, yrs_ in [("2019-2021", [2019, 2020, 2021]), ("2022-2023", [2022, 2023])]:
    sub = first[first.t.isin(yrs_)]
    a, b = sub[sub.level == 1], sub[sub.level != 1]
    la, ha = proportion_confint(int(a.out.sum()), len(a), method="wilson")
    lb, hb = proportion_confint(int(b.out.sum()), len(b), method="wilson")
    res["分段"][lab] = {"L1_n": int(len(a)), "L1_rate": float(a.out.mean()), "L1_lo": float(la), "L1_hi": float(ha),
                        "rest_n": int(len(b)), "rest_rate": float(b.out.mean()), "rest_lo": float(lb), "rest_hi": float(hb)}
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(OUT / "attribution.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
comp.to_csv(OUT / "attribution_groups.csv", encoding="utf-8-sig")
chan.to_csv(OUT / "attribution_channels.csv", index=False, encoding="utf-8-sig")

# 图（fig9_attribution）由 23b_attribution_fig.py 从上面保存的结果绘制
