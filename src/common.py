"""公共工具：路径、读缓存、名称规范化、案件分类、时间常量。"""
import re, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
EXT = CACHE / "ext"
WORK = CACHE / "work"
OUT = ROOT / "output"
for p in (WORK, OUT):
    p.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 观察年 t：决策点 D_t = (t+1)-04-30（FY t 年报已披露），标签窗 (D_t, D_t+12m]
YEARS = list(range(2016, 2024))
TRAIN_YEARS, VALID_YEARS, TEST_YEARS = [2016, 2017, 2018, 2019, 2020], [2021], [2022, 2023]
HALF_LIFE_M = 12      # 风险事件时间衰减半衰期（月）
LOOKBACK_M = 36       # 风险事件回看窗口（月）
EXCLUDE_M = 24        # 过去 24 个月已发生债务违约的企业不进入样本（只预测"首次"违约）


def decision_date(t: int) -> pd.Timestamp:
    return pd.Timestamp(f"{t + 1}-04-30")


def load_raw(name: str, skip: int) -> pd.DataFrame:
    """读 00_convert 生成的缓存，去掉中文表头行（供应链表还有单位行）。"""
    return pd.read_parquet(CACHE / f"{name}.parquet").iloc[skip:].reset_index(drop=True)


def num(s):
    return pd.to_numeric(s, errors="coerce")


_PAREN = str.maketrans({"（": "(", "）": ")", "【": "(", "】": ")", "[": "(", "]": ")"})


def norm_name(x) -> str:
    """企业名称规范化：全角转半角、去空白、统一括号。只做等价变换，绝不做子串截断。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x)).translate(_PAREN)
    s = re.sub(r"\s+", "", s).strip(".。,，;；:：-_ ")
    s = re.sub(r"(等|等公司|等单位)$", "", s)
    return s


_ANON = re.compile(
    r"(\*|＊|某|XX|xx|××|未披露|不适用|无$|^其他|^其它|合计|小计|总计|前五|前5|^第?[一二三四五六七八九十0-9]+名?$"
    r"|^(境外|国内|海外)?(供应商|客户|单位|法人|公司|厂商|企业|用户|关联方|销售客户|采购供应商|客商|经销商)"
    r"[\s\-_]*[A-Za-z0-9一二三四五六七八九十甲乙丙丁戊己庚辛壬癸]+$"
    r"|^[A-Za-z]{1,3}(公司|客户|单位|集团|企业|供应商)?$|^(客户|供应商)(一|二|三|四|五|[A-E1-5])$)"
)


def is_anonymous(name: str) -> bool:
    return (len(name) < 4) or bool(_ANON.search(name))


# 金融机构/产品/名义持有人：作为股东时不代表控制关系，建股权边时剔除
_FIN_HOLDER = re.compile(
    r"(香港中央结算|中央结算|HKSCC|NOMINEES|证券股份|证券有限|证券公司|证券投资|证券\(香港\)|SECURITIES|国际金融股份|"
    r"基金|资产管理|资本管理有限公司|资管|保险|社保|社会保障|养老金|年金|信托|理财|私募|投资管理.*-|"
    r"-.*(计划|产品|组合|专户|账户|资金)|QFII|UBS|MORGAN|GOLDMAN|高盛|J\.P|JPMORGAN|BARCLAYS|MERRILL|CITIGROUP|CITIBANK|"
    r"NOMURA|BNP|HSBC|DEUTSCHE|CREDIT SUISSE|VANGUARD|NORGES|TEMASEK|淡马锡|法国兴业|投资局|政府投资|主权|"
    r"银行股份有限公司-|中国证券金融|中央汇金|汇金|国新投资|金融资产投资|资本运营|创业投资|创新投资|高新投|股权投资|"
    r"产业投资基金|投资中心\(有限合伙\)|员工持股|持股计划|股权激励|限售|回购专用|证券账户)", re.I)
_GOV = re.compile(r"(人民政府|国资委|国有资产监督管理|财政局|财政厅|财政部|国务院)")


def is_gov(name: str) -> bool:
    return bool(_GOV.search(name))


def is_fin_holder(name: str) -> bool:
    return bool(_FIN_HOLDER.search(name))


_PERSON = re.compile(r"^[\u4e00-\u9fa5·]{2,4}$")


def is_person(name: str) -> bool:
    return bool(_PERSON.match(name)) and not re.search(r"(公司|集团|中心|局|厅|院|所|会|部|委|行|社|厂|场|站|队|处)$", name)


# 案件类型：fin=金融债务（借款/票据/融资租赁/担保/债券），trade=经营性欠款
FIN_PAT = re.compile(r"借款|贷款|借贷|票据|融资租赁|保证合同|保证纠纷|担保|追偿|债券|金融|保理|信用证")
TRADE_PAT = re.compile(r"欠款|货款|拖欠|债务|债权|工程款|买卖合同|建设工程|施工合同|加工合同|承揽|采购合同|供货")


def case_type(cname: str) -> str:
    c = cname or ""
    if FIN_PAT.search(c):
        return "fin"
    if TRADE_PAT.search(c):
        return "trade"
    return "other"


SPLIT = re.compile(r"[,，、;；/]|(?<=公司)和|(?<=公司)及")


def split_parties(s) -> list:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    out = []
    for p in SPLIT.split(str(s)):
        n = norm_name(p)
        n = re.sub(r"^(原告|被告|申请人|被申请人|第三人)[一二三四五六七八九十0-9]*[:：]?", "", n)
        if len(n) >= 2:
            out.append(n)
    return out


def decay(months_ago, half_life=HALF_LIFE_M):
    return np.power(0.5, np.asarray(months_ago, dtype=float) / half_life)


SW_L1 = {"11": "农林牧渔", "21": "采掘", "22": "基础化工", "23": "钢铁", "24": "有色金属", "27": "电子", "28": "汽车",
         "33": "家用电器", "34": "食品饮料", "35": "纺织服饰", "36": "轻工制造", "37": "医药生物", "41": "公用事业",
         "42": "交通运输", "43": "房地产", "45": "商贸零售", "46": "社会服务", "48": "银行", "49": "非银金融",
         "51": "综合", "61": "建筑材料", "62": "建筑装饰", "63": "电力设备", "64": "机械设备", "65": "国防军工",
         "71": "计算机", "72": "传媒", "73": "通信", "74": "煤炭", "75": "石油石化", "76": "环保", "77": "美容护理"}


# 特征中文名（作图与产品原型共用）
FEAT_CN = {
    "impair_to_rev": "资产减值/营收", "sw_l1_c": "申万一级行业", "fund_n": "持股基金家数", "cfo_to_due": "现金流到期债务保障",
    "firm_age": "上市年限", "cash_ratio": "现金比率", "tang_lev": "有形资产负债率", "top1_stake": "第一大股东持股",
    "log_ta": "总资产(对数)", "fin_exp_ratio": "财务费用率", "wc_to_loan": "营运资金/借款", "d_roa": "ROA变动",
    "lev": "资产负债率", "d_cfo_to_debt": "现金债务比变动", "inst_ratio": "机构持股比例", "cfo_to_debt": "现金债务总额比",
    "int_debt_ratio": "带息负债比率", "quick_ratio": "速动比率", "cur_ratio": "流动比率", "rel_rev_growth": "相对同业营收增速",
    "pagerank": "PageRank中心度", "period_exp_ratio": "期间费用率", "nb_codef_n": "共同被告数", "own_viol": "违规处罚(衰减)",
    "ltloan_to_ta": "长期借款/总资产", "roa": "ROA", "roe": "ROE", "gross_margin": "毛利率", "net_margin": "净利率",
    "own_trade": "经营性欠款被诉", "own_other_def": "其他案件被诉", "own_def_n12": "近12月被诉次数", "own_def_amt12": "近12月被诉金额",
    "hop1_all": "一跳关联风险", "hop2_all": "二跳关联风险", "ppr3": "三跳扩散风险", "deg": "关联度数", "deg_listed": "上市关联方数",
    "kcore": "k-core", "group_size": "控制圈规模", "group_fin": "控制圈违约暴露", "group_risk": "控制圈风险", "peer_fin_rate": "同业违约率",
    "peer_roa": "同业ROA", "rel_roa": "相对同业ROA", "peer_rev_growth": "同业营收增速", "rel_d_roa": "相对同业ROA变动",
    "log_rev": "营业收入(对数)", "rev_growth": "营收增速", "eq_mult": "权益乘数", "int_cover": "利息保障倍数", "cfo_to_cl": "现金流动负债比",
    "cfo_to_intdebt": "现金带息债务比", "ebit_margin": "息税前利润率", "cfo_to_profit": "现金/利润总额", "d_lev": "负债率变动",
    "d_gross_margin": "毛利率变动", "d_cur_ratio": "流动比率变动", "loss": "当年亏损", "loss_2y": "连续两年亏损", "own_probe12": "近12月立案调查",
    "sell_major": "大股东净减持", "sell_exec": "高管净减持", "sell_person": "个人股东净减持", "top1_person": "第一大股东为自然人",
    "soe": "国有控股", "top10_nonfin": "前十大非金融股东持股", "board_c": "上市板块", "own_fin_old": "24-36月前违约",
    "own_sec_fraud": "虚假陈述被诉", "own_viol_severe": "严重违规处罚", "peer_n": "同业家数", "peer_loss_rate": "同业亏损率",
    "peer_d_roa": "同业ROA变动", "group_listed": "控制圈上市公司数", "grs": "图谱风险分GRS",
    "rev_growth": "营收增速", "net_margin": "净利率", "ebit_margin": "息税前利润率", "own_def_amt12": "近12月被诉金额", "own_def_n12": "近12月被诉次数",
    "own_fin_old": "24-36月前违约", "hop1_all": "一跳关联风险", "hop2_all": "二跳关联风险",
}
