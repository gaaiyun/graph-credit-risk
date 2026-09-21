"""赛题业务背景中三条机制的专项检验 + 动态事件乘数。
A. 年度截面（线性概率模型，行业×年份固定效应，企业聚类标准误；控制与 16 号脚本相同的自身变量）：
   ① 上游供应商提价：毛利率变动（挤压为负）× 前五大供应商集中度
   ② 投资方的其他被投企业出险：共同控股股东/同一控制的兄弟上市公司、共同创投/产业基金的其他被投上市公司，决策日前 12 个月出险
   ③ 竞争对手突破：同业（申万二级，剔除自身）营收增速超过 30% 的企业占比 × 本企业营收增速落后同业
B. 动态事件乘数（月度样本外）：以样本外 D1 评分的对数几率为 offset，估计近 3 个月各类关联事件对未来 6 个月违约几率的增量乘数。"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import *

P = pd.read_parquet(WORK / "panel.parquet")
DY = pd.read_parquet(WORK / "dyn_panel.parquet")
fin = pd.read_parquet(WORK / "fin_panel.parquet")
conc = pd.read_parquet(WORK / "sc_conc.parquet")
sw = pd.read_parquet(WORK / "sw_hist.parquet")

D0 = DY[DY.k == 0][["code", "t", "dy_sib_ctrl_12m", "dy_sib_pe_12m", "sib_ctrl_n", "sib_pe_n"]]
X = P.merge(D0, on=["code", "t"], how="left").merge(conc.rename(columns={"fy": "t"}), on=["code", "t"], how="left")
# 同业高速增长者占比（剔除自身）
rows = []
for t in YEARS:
    D = decision_date(t)
    ind = sw[sw.start <= D].groupby("code").sw.last().str[:4]
    f = fin[fin.fy == t][["code", "rev_growth"]].copy()
    f["l2"] = f.code.map(ind)
    f = f.dropna(subset=["l2"])
    f["fast"] = (f.rev_growth > 0.3).astype(float)
    g = f.groupby("l2").agg(n=("fast", "size"), fast=("fast", "sum"))
    f = f.join(g, on="l2", rsuffix="_g")
    f["peer_fast_share"] = (f.fast_g - f.fast) / (f.n - 1).clip(lower=1)
    rows.append(f[["code", "peer_fast_share"]].assign(t=t))
X = X.merge(pd.concat(rows), on=["code", "t"], how="left")

ctrl = ["lev", "roa", "cfo_to_debt", "cash_ratio", "log_ta", "impair_to_rev", "own_trade", "own_other_def", "own_viol", "firm_age", "soe"]


def std(s):
    s = s.astype(float)
    s = s.clip(s.quantile(0.01), s.quantile(0.99))
    s = s.fillna(s.median())
    return (s - s.mean()) / s.std()


cell = X.sw_l1.astype(str) + "_" + X.t.astype(str)
base = X.y.mean()


def lpm(terms, label):
    Z = pd.concat([pd.DataFrame({k: v for k, v in terms.items()}), pd.DataFrame({c: std(X[c]) for c in ctrl})], axis=1)
    Zd = Z - Z.groupby(cell.values).transform("mean")
    yd = X.y - X.y.groupby(cell.values).transform("mean")
    r = sm.OLS(yd.values, Zd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(X.code)[0]})
    out = []
    for i, k in enumerate(terms):
        b, se = r.params[i], r.bse[i]
        out.append({"机制": label, "变量": k, "系数(百分点)": 100 * b, "CI_lo": 100 * (b - 1.96 * se), "CI_hi": 100 * (b + 1.96 * se),
                    "相对基准": b / base, "覆盖": float((terms[k] != 0).mean()) if terms[k].nunique() <= 2 else np.nan})
    return out


res = []
# ① 上游供应商提价：毛利率挤压（标准化后取负值，越大越挤压）× 高供应商集中度
squeeze = -std(X.d_gross_margin)
hi_sup = (X.sup_conc >= X.sup_conc.median()).astype(float).where(X.sup_conc.notna(), 0.0)
res += lpm({"毛利率挤压(1SD)": squeeze, "供应商集中度高": hi_sup, "挤压×集中度高": squeeze * hi_sup}, "①上游供应商提价")
# ② 投资方其他被投企业出险
res += lpm({"兄弟企业出险(共同控股股东/同一控制)": (X.dy_sib_ctrl_12m.fillna(0) > 0).astype(float),
            "其他被投企业出险(共同创投/产业基金)": (X.dy_sib_pe_12m.fillna(0) > 0).astype(float)}, "②投资方其他被投项目出险")
# ③ 竞争对手突破
lag_ = (X.rel_rev_growth < 0).astype(float)
fast = std(X.peer_fast_share)
res += lpm({"同业高速增长者占比(1SD)": fast, "本企业增速落后同业": lag_, "高速增长者占比×落后": fast * lag_}, "③竞争对手突破")
mech = pd.DataFrame(res)
mech.to_csv(OUT / "mechanisms.csv", index=False, encoding="utf-8-sig")
print(f"基准违约率 {base:.4f}")
print(mech.round(3).to_string(index=False))

# B. 动态事件乘数：样本外 D1 评分为 offset
R = pd.read_parquet(WORK / "dyn_oot.parquet")
d1 = [c for c in R.columns if c.startswith("D1 ")][0]
R = R.merge(DY[["code", "t", "k"] + [c for c in DY.columns if c.endswith("_3m") and c.startswith("dy_")]], on=["code", "t", "k"], how="left")
lab = {"dy_holder_3m": "股东", "dy_codef_3m": "共同被告(担保连带)", "dy_debtor_3m": "债务人", "dy_cus_3m": "客户", "dy_sup_3m": "供应商",
       "dy_ctrl_3m": "同一控制关联方", "dy_sib_ctrl_3m": "兄弟企业(共同控股股东)", "dy_sib_pe_3m": "其他被投企业(共同创投)",
       "dy_sc2hop_3m": "客户的客户/供应商的供应商", "dy_all_enf_3m": "任一关联方被强制执行/失信"}
Z = pd.DataFrame({v: (R[k].fillna(0) > 0).astype(float) for k, v in lab.items()})
p = R[d1].clip(1e-6, 1 - 1e-6)
off = np.log(p / (1 - p))
glm = sm.GLM(R.y6.values, sm.add_constant(Z), family=sm.families.Binomial(), offset=off.values).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(R.code)[0]})
mult = []
for v in lab.values():
    b, se = glm.params[v], glm.bse[v]
    mult.append({"关联事件(近3个月)": v, "事件乘数": float(np.exp(b)), "CI_lo": float(np.exp(b - 1.96 * se)), "CI_hi": float(np.exp(b + 1.96 * se)),
                 "企业-月覆盖": float(Z[v].mean())})
mult = pd.DataFrame(mult)
mult.to_csv(OUT / "dyn_multipliers.csv", index=False, encoding="utf-8-sig")
print(mult.round(3).to_string(index=False))
