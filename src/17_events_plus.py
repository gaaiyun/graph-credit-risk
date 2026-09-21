"""统一的带日期风险事件表（所有节点），供月度动态模型使用：
- fin / trade：作为被告卷入金融债务 / 经营性欠款诉讼（本公司披露 + 被其他上市公司列为被告）
- enforce：诉讼"执行情况"中出现强制执行、被执行、查封、冻结、拍卖、终结本次执行、失信（取执行情况记录的日期）
- viol：违规处罚（公司本身、公司股东）
另输出供应链集中度（前五大供应商/客户占比与 HHI）。"""
import re
import numpy as np
import pandas as pd
from common import *

cases = pd.read_parquet(WORK / "lit_cases.parquet")
parties = pd.read_parquet(WORK / "lit_parties.parquet")
vio = pd.read_parquet(WORK / "violations.parquet")
nd = pd.read_parquet(WORK / "name_dict.parquet")

rows = []
# 本公司为被告（含子公司等）的金融/欠款案件
own = cases[cases.self_def & cases.ctype.isin(["fin", "trade"])]
rows.append(pd.DataFrame({"node": "C:" + own.code, "date": own.date, "kind": own.ctype, "case_id": own.case_id}))
# 其他当事人（被告）
p = parties[(parties.side == "def") & parties.ctype.isin(["fin", "trade"])]
p = p[p.name_code.isna() | (p.name_code != p.code)]
rows.append(pd.DataFrame({"node": np.where(p.name_code.notna(), "C:" + p.name_code.astype(str), "E:" + p.name),
                          "date": p.date, "kind": p.ctype, "case_id": p.case_id}))

# 执行 / 失信：解析执行情况的日期前缀
raw = load_raw("上市公司诉讼仲裁统计", 1)
raw["case_id"] = np.arange(len(raw))
ex = raw[["case_id", "Exctcdt"]].dropna()
ENF = re.compile(r"强制执行|被执行|查封|冻结|拍卖|终结本次执行|失信|限制高消费|限高")
ex = ex[ex.Exctcdt.str.contains(ENF)]
m = ex.Exctcdt.str.extract(r"^\s*(\d{4})[.\-/]?(\d{2})[.\-/]?(\d{2})")
ex["date"] = pd.to_datetime(m[0] + "-" + m[1] + "-" + m[2], errors="coerce")
ex["shixin"] = ex.Exctcdt.str.contains("失信|限制高消费|限高")
ex = ex.dropna(subset=["date"])
ex = ex.merge(cases[["case_id", "code", "self_def", "ctype"]], on="case_id")
# 执行对象：本公司为被告则为本公司，否则为该案被告方
e_own = ex[ex.self_def]
rows.append(pd.DataFrame({"node": "C:" + e_own.code, "date": e_own.date, "kind": np.where(e_own.shixin, "shixin", "enforce"),
                          "case_id": e_own.case_id}))
e_oth = ex[~ex.self_def].merge(parties[parties.side == "def"][["case_id", "name", "name_code"]], on="case_id")
e_oth = e_oth[e_oth.name_code.isna() | (e_oth.name_code != e_oth.code)]
rows.append(pd.DataFrame({"node": np.where(e_oth.name_code.notna(), "C:" + e_oth.name_code.astype(str), "E:" + e_oth.name),
                          "date": e_oth.date, "kind": np.where(e_oth.shixin, "shixin", "enforce"), "case_id": e_oth.case_id}))
# 违规处罚
v_own = vio[vio.Relationship == "公司本身"]
rows.append(pd.DataFrame({"node": "C:" + v_own.code, "date": v_own.date, "kind": "viol", "case_id": -1}))
v_sh = vio[vio.Relationship == "公司股东"].merge(nd.rename(columns={"name": "obj_name", "code": "oc"}), on="obj_name", how="left")
rows.append(pd.DataFrame({"node": np.where(v_sh.oc.notna(), "C:" + v_sh.oc.astype(str), "E:" + v_sh.obj_name),
                          "date": v_sh.date, "kind": "viol", "case_id": -1}))
EV = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
EV = EV[EV.node.str.len() > 3].drop_duplicates(["node", "date", "kind", "case_id"])
EV["mi"] = EV.date.dt.year * 12 + EV.date.dt.month - 1           # 月序号
EV.to_parquet(WORK / "events_all.parquet", index=False)
print(EV.kind.value_counts().to_dict(), "节点数", EV.node.nunique())
print("执行/失信事件按年:", EV[EV.kind.isin(["enforce", "shixin"])].date.dt.year.value_counts().sort_index().to_dict())

# 企业作为当事人的案件（用于剔除"同一宗案件"的回声）
pc = pd.concat([cases[["case_id", "code"]], parties[parties.name_code.notna()][["case_id", "name_code"]].rename(columns={"name_code": "code"})])
pc.drop_duplicates().to_parquet(WORK / "party_cases.parquet", index=False)

# 供应链集中度
cc = load_raw("SC_ConcentrationIndex", 2)
cc = cc[cc.StateTypeCode == "1"].copy()
cc["code"] = cc.Symbol.str.zfill(6)
cc["fy"] = cc.EndDate.str[:4].astype(int)
out = pd.DataFrame({"code": cc.code, "fy": cc.fy, "sup_conc": num(cc.PurchaseConcentration) / 100, "cus_conc": num(cc.CustomerConcentration) / 100,
                    "sup_hhi": num(cc.PurchaseConcentrationHHI) / 100, "cus_hhi": num(cc.CustomerConcentrationHHI) / 100})
out.drop_duplicates(["code", "fy"]).to_parquet(WORK / "sc_conc.parquet", index=False)
print("集中度", out.shape, out[["sup_conc", "cus_conc"]].describe().round(3).loc[["mean", "50%"]].to_dict())
