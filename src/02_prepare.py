"""数据清洗与标准化：企业主表、名称字典、事件表（诉讼/违规/减持）、财务面板、各类关系边原始表。
输出到 cache/work/*.parquet，后续建图与特征只读这里。"""
import re
import numpy as np
import pandas as pd
from common import *

# ---------------------------------------------------------------- 1. 企业主表 + 名称字典
sh = pd.concat([pd.read_parquet(EXT / "sh_main.parquet"), pd.read_parquet(EXT / "sh_kcb.parquet")])
sh = sh.rename(columns={"证券代码": "code", "公司全称": "full", "证券简称": "short", "上市日期": "list_date"})
sz = pd.read_excel(EXT / "szse_list_raw.xlsx", dtype=str)
sz.columns = [re.sub(r"\s+", "", c) for c in sz.columns]
sz = sz.rename(columns={"A股代码": "code", "公司全称": "full", "A股简称": "short", "A股上市日期": "list_date",
                        "省份": "province", "所属行业": "csrc_ind"})
bj = pd.read_parquet(EXT / "bj_a.parquet").rename(columns={"证券代码": "code", "证券简称": "short", "上市日期": "list_date", "地区": "province"})
cur = pd.concat([sh[["code", "full", "short", "list_date"]], sz[["code", "full", "short", "list_date", "province"]],
                 bj[["code", "short", "list_date", "province"]]], ignore_index=True)
cur["code"] = cur.code.str.zfill(6)

names = []  # (规范化全称, 代码, 来源)
for r in cur.dropna(subset=["full"]).itertuples():
    names.append((norm_name(r.full), r.code, "exchange"))
# 历史全称：供应链中标注为上市公司的对手方名称；违规处罚中处罚对象=公司本身
for f in ["SC_TopFivePurchaseInfo", "SC_TopFiveSaleInfo"]:
    d = load_raw(f, 2)
    d = d[(d.IsListed == "Y") & d.BusinessSymbol.notna()]
    for n, c in zip(d.InstitutionName, d.BusinessSymbol):
        names.append((norm_name(n), str(c).zfill(6), "sc"))
vio = load_raw("上市公司违规与处罚", 1)
d = vio[(vio.Relationship == "公司本身") & vio.PunishObj.fillna("").str.endswith("股份有限公司")]
for n, c in zip(d.PunishObj, d.SCode):
    names.append((norm_name(n), str(c).zfill(6), "vio"))
nd = pd.DataFrame(names, columns=["name", "code", "src"])
nd = nd[(nd.name.str.len() >= 6) & nd.code.str.match(r"^(0|3|6|8|4|9)\d{5}$")]
# 一个名称对应多个代码（A/B 股、借壳换码等）时取出现次数最多者
nd = (nd.groupby(["name", "code"]).size().rename("n").reset_index()
        .sort_values(["name", "n"], ascending=[True, False]).drop_duplicates("name"))
nd[["name", "code"]].to_parquet(WORK / "name_dict.parquet", index=False)
print(f"[name_dict] {len(nd)} 个全称 -> {nd.code.nunique()} 个代码")

# 申万行业历史（按计入日期生效）
sw = pd.read_excel(EXT / "sw_StockClassifyUse_stock.xls", dtype=str)
sw = sw.rename(columns={"股票代码": "code", "计入日期": "start", "行业代码": "sw"})
sw["start"] = pd.to_datetime(sw.start)
sw = sw[["code", "start", "sw"]].sort_values(["code", "start"])
sw.to_parquet(WORK / "sw_hist.parquet", index=False)

# 上市日期：交易所列表优先，退市公司用退市名单，再用申万首次计入日期兜底
dl_sh = pd.read_parquet(EXT / "sh_delist.parquet").rename(columns={"公司代码": "code", "公司简称": "short", "上市日期": "list_date", "暂停上市日期": "delist_date"})
dl_sz = pd.read_parquet(EXT / "sz_delist.parquet").rename(columns={"证券代码": "code", "证券简称": "short", "上市日期": "list_date", "终止上市日期": "delist_date"})
dl = pd.concat([dl_sh, dl_sz])[["code", "short", "list_date", "delist_date"]]
dl["code"] = dl.code.str.zfill(6)
master = pd.concat([cur[["code", "short", "list_date", "province"]], dl], ignore_index=True)
master["list_date"] = pd.to_datetime(master.list_date, errors="coerce")
master["delist_date"] = pd.to_datetime(master.get("delist_date"), errors="coerce")
master = master.sort_values("list_date").groupby("code").agg(
    short=("short", "last"), list_date=("list_date", "first"), delist_date=("delist_date", "max"),
    province=("province", "first")).reset_index()
sw_first = sw.groupby("code").start.min().rename("sw_first")
master = master.merge(sw_first, on="code", how="outer")
master["list_date"] = master.list_date.fillna(master.sw_first)
full_latest = nd.groupby("code").name.first().rename("full")
master = master.merge(full_latest, on="code", how="left").drop(columns="sw_first")
master["board"] = np.select([master.code.str[:3].isin(["300", "301"]), master.code.str[:3].isin(["688", "689"]),
                             master.code.str[:1].isin(["8", "4"]) | master.code.str[:3].eq("920")],
                            ["创业板", "科创板", "北交所"], "主板")
master.to_parquet(WORK / "firm_master.parquet", index=False)
print(f"[firm_master] {len(master)} 家；有全称 {master.full.notna().sum()}；有上市日期 {master.list_date.notna().sum()}")

# ---------------------------------------------------------------- 2. 诉讼事件（按当事人展开）
lit = load_raw("上市公司诉讼仲裁统计", 1)
lit["code"] = lit.Scode.str.zfill(6)
lit["date"] = pd.to_datetime(lit.Deldt, errors="coerce")
lit["amt"] = num(lit.Amount)
lit["ctype"] = lit.Cname.fillna("").map(case_type)
role = lit.Clitype.fillna("")
lit["self_def"] = role.isin(["被告", "被告（子公司）", "被告（孙公司）", "被告（分公司）"])
lit["self_pla"] = role.isin(["原告", "原告（子公司）", "原告（孙公司）", "原告（分公司）"])
lit["holder_def"] = role.isin(["被告（股东）", "被告（关联方）", "被告（联营公司）"])  # 股东/关联方被诉：传导信号
lit["case_id"] = np.arange(len(lit))
lit[["case_id", "code", "date", "ctype", "amt", "Clitype", "Cname", "self_def", "self_pla", "holder_def",
     "Prstn", "Defense", "Lstype"]].to_parquet(WORK / "lit_cases.parquet", index=False)

rows = []
for r in lit[["case_id", "code", "date", "ctype", "Prstn", "Defense"]].itertuples(index=False):
    for p in split_parties(r.Defense):
        rows.append((r.case_id, r.code, r.date, r.ctype, "def", p))
    for p in split_parties(r.Prstn):
        rows.append((r.case_id, r.code, r.date, r.ctype, "pla", p))
parties = pd.DataFrame(rows, columns=["case_id", "code", "date", "ctype", "side", "name"])
parties = parties.merge(nd[["name", "code"]].rename(columns={"code": "name_code"}), on="name", how="left")
parties.to_parquet(WORK / "lit_parties.parquet", index=False)
fin_def = lit.self_def & (lit.ctype == "fin")
print(f"[lit] 案件 {len(lit)}；当事人记录 {len(parties)}；可对齐到上市公司的当事人 {parties.name_code.notna().sum()}")
print(f"      本公司为被告的金融债务案件 {int(fin_def.sum())}，涉及 {lit[fin_def].code.nunique()} 家")

# ---------------------------------------------------------------- 3. 违规处罚
vio["code"] = vio.SCode.str.zfill(6)
vio["date"] = pd.to_datetime(vio.AnncDate, errors="coerce")
vio["penalty"] = num(vio.Penalty)
disp = vio.DispTp.fillna("")
vio["severe"] = disp.str.contains("立案调查|公开谴责|公开处罚|警告|罚款|市场禁入|没收").astype(int)
vio["obj_name"] = vio.PunishObj.map(norm_name)
vio[["code", "date", "Relationship", "severe", "penalty", "obj_name", "DispTp", "VioAct"]].to_parquet(WORK / "violations.parquet", index=False)

# ---------------------------------------------------------------- 4. 股东减持（重要股权交易，含高管与大股东）
tr = load_raw("重要股权交易", 1)
tr["code"] = tr.SCode.str.zfill(6)
tr["date"] = pd.to_datetime(tr.AnncDate, errors="coerce")
tr["shares"] = num(tr.ChgNbr) * np.where(tr.ChgDir == "减持", 1, -1)   # 正=净减持
tr[["code", "date", "ShrhTp", "shares", "ShrhNm"]].to_parquet(WORK / "trades.parquet", index=False)

inst = load_raw("机构持股", 1)
inst["code"] = inst.SCode.str.zfill(6)
inst["fy"] = inst.Date.str[:4].astype(int)
inst["tot_shares"] = num(inst.TlShrN)
ratio_cols = ["FCSRt", "ICSRt", "SCSRt", "QFIICSRt", "OCSRt", "BCSRt"]
inst["inst_ratio"] = inst[ratio_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
inst["fund_n"] = num(inst.FShrhN)
inst[["code", "fy", "tot_shares", "inst_ratio", "fund_n"]].to_parquet(WORK / "inst.parquet", index=False)

# ---------------------------------------------------------------- 5. 年度财务面板（FY 年末）
sol = load_raw("偿债能力", 1)
pro = load_raw("盈利能力", 1)
sol = sol[sol.Date.str[5:] == "12-31"].copy()
pro = pro[pro.Date.str[5:] == "12-31"].copy()
for d in (sol, pro):
    d["code"] = d.Scode.str.zfill(6)
    d["fy"] = d.Date.str[:4].astype(int)
sol_cols = {"Curtrt": "cur_ratio", "Qikrt": "quick_ratio", "Cashrt": "cash_ratio", "Wkcpt_bw": "wc_to_loan",
            "Lwirt": "int_debt_ratio", "Aslbrt": "lev", "Equmlt": "eq_mult", "Notie": "int_cover", "Cdcrt": "cfo_to_debt",
            "Opncf_cl": "cfo_to_cl", "Opncf_lwi": "cfo_to_intdebt", "Opncf_ncldoynp": "cfo_to_due", "Tgasdbrt": "tang_lev",
            "Ltbl_at": "ltloan_to_ta", "Ebit": "ebit"}
pro_cols = {"Roa_1": "roa", "Roe_1": "roe", "Salgm": "gross_margin", "Salnpm": "net_margin", "Fiexprt": "fin_exp_ratio",
            "Ail_rev": "impair_to_rev", "Opncf_ibt": "cfo_to_profit", "Ebitpm": "ebit_margin", "Selprdert": "period_exp_ratio"}
fin = sol[["code", "fy"] + list(sol_cols)].rename(columns=sol_cols).merge(
      pro[["code", "fy"] + list(pro_cols)].rename(columns=pro_cols), on=["code", "fy"], how="outer")
for c in list(sol_cols.values()) + list(pro_cols.values()):
    fin[c] = num(fin[c])
fin = fin.drop_duplicates(["code", "fy"])
# 规模：营业收入 ≈ EBIT / 息税前营业利润率；总资产 ≈ 净利润 / ROA（分母过小时置空）
rev = fin.ebit / (fin.ebit_margin / 100)
rev = rev.where((fin.ebit_margin.abs() > 0.5) & (rev > 0))
ta = (fin.net_margin / 100 * rev) / (fin.roa / 100)
ta = ta.where((fin.roa.abs() > 0.2) & (ta > 0))
fin["log_rev"] = np.log(rev)
fin["log_ta"] = np.log(ta)
fin = fin.sort_values(["code", "fy"])
g = fin.groupby("code")
for c in ["roa", "lev", "gross_margin", "cfo_to_debt", "cur_ratio"]:
    fin[f"d_{c}"] = fin[c] - g[c].shift(1)
fin["rev_growth"] = fin.log_rev - g.log_rev.shift(1)
fin["loss"] = (fin.roa < 0).astype(float)
fin["loss_2y"] = ((fin.roa < 0) & (g.roa.shift(1) < 0)).astype(float)
fin.to_parquet(WORK / "fin_panel.parquet", index=False)
print(f"[fin] {fin.shape}  log_rev 覆盖 {fin.log_rev.notna().mean():.2%}  log_ta 覆盖 {fin.log_ta.notna().mean():.2%}")

# ---------------------------------------------------------------- 6. 供应链边（前五大客户/供应商，剔除匿名）
sc = []
for f, rel in [("SC_TopFivePurchaseInfo", "sup"), ("SC_TopFiveSaleInfo", "cus")]:
    d = load_raw(f, 2)
    d = d[(d.StateTypeCode == "1") & (d.Rank != "6")].copy()
    d["code"] = d.Symbol.str.zfill(6)
    d["fy"] = d.EndDate.str[:4].astype(int)
    d["name"] = d.InstitutionName.map(norm_name)
    d["share"] = num(d.ProportionOfTotalValue) / 100
    d["listed_code"] = np.where(d.IsListed == "Y", d.BusinessSymbol.fillna("").str.zfill(6), None)
    d["rel"] = rel
    d["related"] = d.IsRelatedCompany.notna()
    sc.append(d[["code", "fy", "rel", "name", "share", "listed_code", "related"]])
sc = pd.concat(sc, ignore_index=True)
sc["anon"] = sc.name.map(is_anonymous)
sc = sc.merge(nd.rename(columns={"code": "name_code"}), on="name", how="left")
lc = sc.listed_code.fillna("")
sc["nb_code"] = np.where(lc.str.len() == 6, lc, sc.name_code)
sc["nb_code"] = sc.nb_code.replace("", np.nan)
sc.to_parquet(WORK / "sc_edges.parquet", index=False)
print(f"[sc] {len(sc)} 条；非匿名 {(~sc.anon).mean():.1%}；对手方为上市公司 {sc.nb_code.notna().sum()}")

# ---------------------------------------------------------------- 7. 股东边（前十大，剔除金融名义持有人）
shr = load_raw("主要股东持股", 1)
shr["code"] = shr.SCode.str.zfill(6)
shr["fy"] = shr.Date.str[:4].astype(int)
shr["name"] = shr.ShrName.map(norm_name)
shr["stake"] = num(shr.ShrRt)
shr["rank"] = num(shr.Num)
shr["fin_holder"] = shr.name.map(is_fin_holder)
shr["person"] = shr.name.map(is_person)
shr = shr.merge(nd.rename(columns={"code": "name_code"}), on="name", how="left")
shr[["code", "fy", "name", "stake", "rank", "fin_holder", "person", "name_code"]].to_parquet(WORK / "shr_edges.parquet", index=False)
print(f"[shr] {len(shr)}；非金融持有人 {(~shr.fin_holder).mean():.1%}；股东为上市公司 {shr.name_code.notna().sum()}")

# ---------------------------------------------------------------- 8. 关联方边
rp = load_raw("关联交易", 1)
rp["code"] = rp.SCode.str.zfill(6)
rp["date"] = pd.to_datetime(rp.AnncDate, errors="coerce")
rp["name"] = rp.RlatParty.map(norm_name).str.replace(r"(及其|及下属|及其下属|及其控制|下属).*$", "", regex=True)
rs = rp.Rlatship.fillna("")
rp["rtype"] = np.select(
    [rs.str.contains("实际控制人|控股股东|母公司|同一集团|同一控制|同一母公司|同一控股|股东子公司|母公司的"),
     rs.str.contains("关键人员|董事|监事|高管|高级管理|关键管理|亲属|家庭成员"),
     rs.str.contains("联营|合营|参股|子公司|孙公司|合资")],
    ["ctrl", "person", "affil"], "other")
rp["amt"] = num(rp.Sum)
rp = rp.merge(nd.rename(columns={"code": "name_code"}), on="name", how="left")
rp = rp[rp.name.str.len() >= 4]
rp[["code", "date", "name", "rtype", "amt", "name_code"]].to_parquet(WORK / "rp_edges.parquet", index=False)
print(f"[rp] {len(rp)}；类型分布 {rp.rtype.value_counts().to_dict()}；关联方为上市公司 {rp.name_code.notna().sum()}")
