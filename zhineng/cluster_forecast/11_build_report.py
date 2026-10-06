"""Step 11: plain-language project report (Chinese) -> outputs/project_report_zh.docx.
All numbers are read from the result files."""
import importlib
import json

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from config import CACHE, FIG, LOCAL_TZ, OUT, SHORT_FIXED, SURPLUS_FIXED, TEST_END, TRAIN_END, VAL_START

SL = FIG / "slides"
CJK = "Microsoft YaHei"
BLUE, DARK, GREY = RGBColor(0x24, 0x71, 0xa3), RGBColor(0x1f, 0x2a, 0x36), RGBColor(0x6b, 0x72, 0x7a)


# ------------------------------------------------------------------ helpers
def set_cjk(style_or_run, latin="Calibri"):
    style_or_run.font.name = latin
    rpr = style_or_run.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts"); rpr.append(fonts)
    fonts.set(qn("w:eastAsia"), CJK)


def para(doc, text="", bold_parts=(), size=10.5, color=None, align=None, space_after=6, italic=False):
    """text may contain **bold** segments."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if align:
        p.alignment = align
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        r = p.add_run(seg)
        set_cjk(r)
        r.font.size, r.bold, r.italic = Pt(size), i % 2 == 1, italic
        if i % 2 == 1:
            r.font.color.rgb = BLUE
        elif color:
            r.font.color.rgb = color
    return p


def bullets(doc, items):
    for it in items:
        p = para(doc, it, space_after=3)
        p.style = doc.styles["List Bullet"]


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        set_cjk(r); r.font.color.rgb = DARK if level == 1 else BLUE
    return h


def shade(cell, hex_color):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tcpr.append(shd)


def callout(doc, title, lines, fill="EAF2F8"):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.cell(0, 0); shade(c, fill)
    c.paragraphs[0].text = ""
    r = c.paragraphs[0].add_run(title); set_cjk(r); r.bold = True; r.font.size = Pt(11); r.font.color.rgb = BLUE
    for ln in lines:
        p = c.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        for i, seg in enumerate(ln.split("**")):
            rr = p.add_run(seg); set_cjk(rr); rr.font.size = Pt(10.5); rr.bold = i % 2 == 1
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]; shade(c, "D6E4F0")
        c.paragraphs[0].text = ""
        r = c.paragraphs[0].add_run(h); set_cjk(r); r.bold = True; r.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].paragraphs[0].text = ""
            r = cells[j].paragraphs[0].add_run(str(v)); set_cjk(r); r.font.size = Pt(10)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, caption, width_cm=14):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True      # keep caption on the same page
    para(doc, caption, size=9, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, space_after=10)


def figures_side_by_side(doc, items, width_cm=7.8):
    t = doc.add_table(rows=1, cols=len(items))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, (path, cap) in enumerate(items):
        c = t.cell(0, j)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraphs[0].add_run().add_picture(str(path), width=Cm(width_cm))
        p = c.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(cap); set_cjk(r); r.font.size = Pt(9); r.italic = True; r.font.color.rgb = GREY
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ------------------------------------------------------------------ numbers
def numbers():
    n = importlib.import_module("10_build_pptx").load_numbers()
    k = n["k"]
    cs = pd.read_csv(OUT / f"cluster_summary_K{k}.csv").set_index("Cluster")
    sens = pd.read_csv(OUT / "level3_sensitivity.csv")
    sens = sens[sens["K"] == k].set_index("Short_fixed_EUR_MWh")
    preds = pd.read_parquet(CACHE / "portfolio_predictions.parquet")
    price = pd.read_parquet(CACHE / "da_price_hourly.parquet")["da_price"].reindex(preds.index)
    q = pd.read_parquet(CACHE / f"level3_quantiles_K{k}.parquet")
    lq = q.index.tz_convert(LOCAL_TZ)
    width = 100 * (q["q90"] - q["q10"]) / q["forecast"]
    winter = pd.Series(lq.month, index=q.index).isin([12, 1, 2]).to_numpy()
    night = (lq.hour <= 6)
    midday = (lq.hour >= 10) & (lq.hour <= 16)
    n.update({
        "cs": cs, "sens": sens,
        "mean_load": preds["actual"].mean(), "price_mean": price.mean(), "neg_hours": int((price < 0).sum()),
        "price_max": price.max(), "width_mean": width.mean(),
        "width_winter_night": width[winter & night].mean(), "width_midday": width[midday].mean(),
        "mae": pd.read_csv(OUT / "metrics_by_k.csv").set_index("K").loc[1, "MAE_kWh"],
        "n_val_hh": int(pd.read_csv(OUT / "validation_selection.csv")["n_households"].iloc[0]),
        "overall_pv": pd.read_csv(OUT / "cluster_assignments.csv")["HasPV"].mean(),
    })
    return n


# ------------------------------------------------------------------ report
def main():
    n = numbers()
    k, cs, sens = n["k"], n["cs"], n["sens"]
    pv_c = cs["PV_share"].idxmax()
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2)
    set_cjk(doc.styles["Normal"]); doc.styles["Normal"].font.size = Pt(10.5)

    # ---------- title ----------
    para(doc, "RWTH Hackathon × E.ON「Bringing the Heat」", size=11, color=BLUE, space_after=2)
    t = para(doc, "先分组，再预测？", size=24, space_after=2); t.runs[0].bold = True
    para(doc, "用聚类和不确定性改进热泵家庭的日前电力采购 —— 项目报告", size=14, color=GREY, space_after=14)

    callout(doc, "一页看懂：我们发现了什么", [
        f"1. 我们为 **{n['n_hh']} 户**装有热泵的家庭预测第二天每小时的用电量，基线模型的误差约 **{n['nmae']:.1f}%**。",
        f"2. 只看用电习惯做聚类，就能把装有**光伏板**的家庭自动分到同一组（该组已知标签的 {n['pv_known']} 户 **100%** 有光伏），完全没用到光伏标签。",
        f"3. 但聚类**并没有显著降低采购成本**：在验证期看起来能省 {n['val']:.1f}%，到真正的测试年只省 {n['test']:.1f}%，统计上和 0 没有区别。",
        f"4. 真正有价值的是**预测的不确定性**：我们的预测区间校准良好；当“缺电时补买”的惩罚较重（≥ {n['sig_from']} €/MWh）时，按不确定性调整报量能节省 **{n['sig_lo']:.0f}–{n['sig_hi']:.0f}%** 的不平衡成本。",
    ])

    # ---------- 1 background ----------
    heading(doc, "1  背景：电是怎么买的，为什么要预测？")
    para(doc, "电很难大量储存，所以发电和用电必须时时刻刻相等。E.ON 这样的售电公司要替客户提前买电，"
              "主要分三步：")
    bullets(doc, [
        "**长期合同**：提前几个月甚至一年买下“基本盘”，最便宜。",
        "**日前市场（day-ahead）**：每天中午为第二天的每个小时报量、竞价。这是本项目关注的环节。",
        "**日内市场（intraday）**：当天发现买多了或买少了，再临时补买或卖出，价格通常更不利。",
    ])
    para(doc, "所以，**日前预测越准，临时补救的代价就越小**。难点在于，越来越多家庭装了**热泵**（用电取暖，"
              "冷天用电猛增）和**光伏板**（晴天中午自己发电，从电网买的电骤降），用电变得更“跳”、更难预测。")
    para(doc, f"**数据**：410 户德国家庭 2019–2024 年每 15 分钟的用电记录，全部装有热泵，约一半装有光伏；"
              f"另有 8 个气象站的每小时天气数据。我们把 15 分钟数据合并成每小时，因为日前市场按小时交易。"
              f"为了“算钱”，我们还下载了德国联邦网络局（SMARD）公布的**真实日前电价**。")

    # ---------- 2 approach ----------
    heading(doc, "2  我们的问题和做法")
    para(doc, "核心问题：**先把相似的家庭分成几组、每组单独预测再相加，会不会比直接预测总量更准、更省钱？**")
    para(doc, "在机器学习里，最容易犯的错误是“偷看答案”。我们把时间线像考试一样切成三段，并严格遵守：")
    table(doc, ["阶段", "时间", "用途", "类比"], [
        ["训练", f"至 {VAL_START:%Y-%m-%d}", "学习规律（分组、训练模型）", "做课本习题"],
        ["验证", f"{VAL_START:%Y-%m-%d} – {TRAIN_END:%Y-%m-%d}", "只用来做选择（分几组）", "模拟考"],
        ["测试", f"{TRAIN_END:%Y-%m-%d} – 2024-02-27", "只在最后看一次，汇报成绩", "期末考"],
    ], widths=[2, 4.8, 5.5, 3])
    para(doc, "选完方案后，模型会用训练+验证的全部数据（约前 80%）重新训练，再去考“期末考”（最后约 20%）。"
              "测试年从头到尾没有参与任何决定，所以它的成绩是可信的。")

    # ---------- 3 level 0 ----------
    heading(doc, "3  Level 0：基线预测")
    para(doc, "基线模型直接预测所有家庭的**总用电量**。它能看到的信息只有“前一天午夜之前就已知”的内容：")
    bullets(doc, [
        "**过去的用电**：昨天、前天、上周同一小时的用电，以及昨天的平均用电；",
        "**日历**：几点、星期几、是否周末；",
        "**天气**：当天的气温和日照（我们用实测值代替天气预报，这让结果略偏乐观）。",
    ])
    para(doc, "模型是**梯度提升树（HistGradientBoosting）**：可以理解为很多棵小“决策树”接力，"
              "每一棵专门修正前面几棵犯的错。它训练快、不需要复杂调参，也能学会“越冷用电越多”这类非线性关系。")
    para(doc, f"结果：在测试年里，平均每小时误差约 {n['mae']:.0f} kWh，相对于平均负荷（约 {n['mean_load']:.0f} kWh/h）"
              f"即 **nMAE ≈ {n['nmae']:.1f}%**。下图是冬季一周，虚线（预测）基本贴住了实线（实际）。")
    figure(doc, SL / "s1_forecast_week.png", "图 1  基线模型在测试集中一个冬季周的预测效果", 14.5)

    # ---------- 4 level 1 ----------
    heading(doc, "4  Level 1：给家庭分组（聚类）")
    para(doc, "我们只用**用电习惯**给每户算了 9 个“画像”指标，例如：")
    bullets(doc, [
        "**温度敏感度**：气温每降 1°C，日用电量增加百分之几（热泵越“卖力”，这个值越大）；",
        "**光伏特征**：夏天中午的用电是否明显低于早晚，晴天中午是否比阴天用得更少；",
        "**时段占比**：夜间、早上、中午、晚上的用电各占多少。",
    ])
    para(doc, "然后用 **k-means 聚类**把画像相近的家庭分到一组，就像按口味把客人分桌。"
              f"分几组（K）不是拍脑袋定的：我们试了 2–6 组，在“模拟考”（验证期）上比较哪个 K 的采购成本最低，"
              f"最终选了 **K = {k}**。注意：**光伏标签从头到尾没有喂给聚类**，只在分完组后拿来检验。")
    rows = []
    for c in cs.index:
        kind = "光伏户" if c == pv_c else ("对温度不敏感" if cs.loc[c, "temp_slope_rel"] < 5 else "典型热泵户")
        rows.append([f"C{c}", int(cs.loc[c, "N"]), kind, f"{cs.loc[c, 'temp_slope_rel']:.1f}%",
                     f"{cs.loc[c, 'pv_midday_ratio']:.2f}", f"{cs.loc[c, 'PV_share']:.0%}"])
    table(doc, ["组", "户数", "特点", "每降 1°C 用电增加", "夏季中午/早晚用电比", "实际有光伏比例"], rows,
          widths=[1.2, 1.3, 3, 3, 3.3, 3])
    para(doc, f"最亮眼的结果：**C{pv_c} 组的 {n['pv_known']} 户（有标签的）全部装有光伏**，而全体家庭的光伏比例只有 "
              f"{n['overall_pv']:.0%}。这组家庭夏天中午从电网买的电只有早晚的约 1/4，因为他们在用自家屋顶发的电。")
    figures_side_by_side(doc, [(SL / "s2_cluster_profiles.png", "图 2a  各组夏季一天的用电形状"),
                               (SL / "s2_pv_share.png", "图 2b  各组光伏比例（标签未用于聚类）")],
                         width_cm=8.0)

    # ---------- 5 level 2 ----------
    heading(doc, "5  Level 2：用“钱”来衡量预测好坏")
    para(doc, "误差百分比对售电公司不够直观，于是我们直接算钱。规则很简单：")
    bullets(doc, [
        "按预测量在日前市场买电，价格用**德国真实日前电价**；",
        f"实际用得**比预测多**（缺电）：要在日内市场补买，每 MWh 多付约 {SHORT_FIXED:.0f} € + 电价的 25%；",
        f"实际用得**比预测少**（多买）：多余的电在日内市场贱卖，每 MWh 少收约 {SURPLUS_FIXED:.0f} € + 电价的 25%。",
    ])
    para(doc, "这些额外花费叫**不平衡成本**。日内惩罚的大小是我们根据市场常识设定的假设，"
              "后面会专门检验结论对它是否敏感。德国电价本身很有特点：傍晚最贵，夏季周末中午因为光伏过剩，"
              f"平均价格甚至为负（测试年共有 {n['neg_hours']} 个负电价小时）。")
    figure(doc, FIG / "level2_german_prices.png", "图 3  测试年德国日前电价：按小时（左）和按月×小时（右）", 15.5)
    para(doc, f"基线模型的不平衡成本约为 **每户每年 {n['cost_hh']:.0f} €**，相当于日前购电费用的约 {n['cost_pct']:.0f}%。"
              "那么，聚类能把它降下来吗？")
    table(doc, ["", "验证期（模拟考）", "测试年（期末考）"], [
        [f"聚类（K={k}）相对基线的节省", f"+{n['val']:.1f}%", f"+{n['test']:.1f}%"],
        ["95% 置信区间", f"[{n['val_lo']:+.1f}%, {n['val_hi']:+.1f}%]", f"[{n['test_lo']:+.1f}%, {n['test_hi']:+.1f}%]"],
        ["结论", "显著节省", "与 0 无显著差别"],
    ], widths=[5.5, 5, 5])
    figure(doc, SL / "s3_validation_vs_test.png", "图 4  聚类带来的节省：验证期 vs 测试年（误差棒为 95% 置信区间）", 11)
    para(doc, "置信区间是用 **bootstrap** 算的：把测试年按周打乱、重新抽样上千次，看节省的结果有多稳定。"
              "区间跨过 0，就说明“省钱”可能只是运气。")
    para(doc, "**为什么验证期的好成绩没能延续？** 一是验证期恰好是 2022/23 年能源危机的冬天，用电行为和电价都很反常；"
              f"二是 {n['n_hh']} 户加在一起后，个体差异已经被大量平均掉，分组带来的额外信息有限。")
    para(doc, f"**一个值得强调的细节**：如果我们偷看测试年再挑 K，会发现 K={n['best_test_k']} 能省 {n['best_test']:.1f}%，"
              "并很可能把它当成结论汇报——这就是“考完再选复习方法”的**选择偏差**。严格的三段划分让我们避开了这个陷阱。")

    # ---------- 6 level 3 ----------
    heading(doc, "6  Level 3：预测的不确定性，以及怎么用它省钱")
    para(doc, "一个数字的预测（“明天 12 点用 500 kWh”）没有告诉我们它有多靠谱。我们进一步给出**预测区间**："
              "例如“有 80% 的把握在 450–560 kWh 之间”。做法很朴素——看过去 8 周同一时段模型通常错多少，"
              "就假设明天也会错差不多这么多。")
    para(doc, f"**区间可信吗？** 在测试年里，真实值落在 80% 区间里的比例是 **{n['cov80']:.0%}**，落在 90% 区间里的是 "
              f"**{n['cov90']:.0%}**，几乎和名义值一致，说明这把“尺子”是准的。")
    para(doc, f"**哪里最没把握？** 区间平均宽度约为预测值的 {n['width_mean']:.0f}%。中午前后最宽（约 {n['width_midday']:.0f}%），"
              f"因为光伏发电和热泵启停都集中在这个时段；冬季夜间最窄（约 {n['width_winter_night']:.0f}%），用电很规律。")
    figure(doc, SL / "s4_uncertainty_by_hour.png", "图 5  一天中不同时段的预测不确定性（80% 区间宽度）", 11)
    para(doc, "**怎么用不确定性省钱？** 这就是经典的“**报童问题**”：面包店老板每天要决定烤多少面包——"
              "烤少了丢生意，烤多了浪费。如果“少了”的代价更大，就应该多烤一点。"
              "对应到买电：最优报量不是预测的平均值，而是预测分布的 **τ* 分位数**，"
              "τ* = 缺电惩罚 ÷（缺电惩罚 + 多买惩罚）。")
    s40, s80, s200 = sens.loc[SHORT_FIXED], sens.loc[n["sig_from"]], sens.loc[n["pen_max"]]
    para(doc, f"在我们的基本假设下（缺电 {SHORT_FIXED:.0f} €、多买 {SURPLUS_FIXED:.0f} €，两边差不多），τ* ≈ {s40['Mean_tau_star']:.2f}，"
              f"几乎就是中间值，所以按 τ* 报量**没有额外收益**（{s40['Saving_pct']:+.1f}%）。"
              f"但现实中缺电的代价常常更高。下图把缺电惩罚从 {SHORT_FIXED:.0f} € 提高到 {n['pen_max']} €/MWh：")
    figure(doc, SL / "s4_value_of_uncertainty.png", "图 6  缺电惩罚越重，按不确定性报量越省钱（阴影为 95% 置信区间）", 12)
    para(doc, f"当缺电惩罚达到 {n['sig_from']} €/MWh（τ* ≈ {s80['Mean_tau_star']:.2f}）时，节省 **{s80['Saving_pct']:.1f}%** 且统计显著；"
              f"到 {n['pen_max']} €/MWh（τ* ≈ {s200['Mean_tau_star']:.2f}）时节省 **{s200['Saving_pct']:.1f}%**。"
              "这远远超过聚类带来的那一点点改善。")

    # ---------- 7 conclusion ----------
    heading(doc, "7  结论：给 E.ON 的三点启示")
    bullets(doc, [
        "**聚类适合用来“认识客户”**：它能不靠标签找出光伏户、划分客户群体；但对几百户以上的大组合，它不是降低日前采购成本的有效手段。",
        "**值得投入的是概率预测**：给出校准良好的区间，并按缺电/多买的成本不对称来调整报量，节省可达两位数百分比。",
        "**评估要像花真金白银一样严格**：按时间三段划分、用真实电价、给出置信区间，才能避免过度承诺。",
    ])
    para(doc, "下一步可以用真实的不平衡结算价格（reBAP）替代我们假设的惩罚、用真实天气预报替代实测天气，"
              "并适配 2025 年 10 月起欧洲日前市场的 15 分钟交易单位。")

    # ---------- glossary ----------
    heading(doc, "术语表")
    table(doc, ["术语", "通俗解释"], [
        ["日前市场（day-ahead）", "前一天为第二天每个小时买电的市场，本项目的预测就是为它服务。"],
        ["日内市场（intraday）", "当天临时补买或卖出电力的市场，通常价格更不利。"],
        ["不平衡成本", "因为预测不准而在日内市场多花的钱（和完美预测相比）。"],
        ["热泵", "用电驱动的取暖设备，天越冷耗电越多。"],
        ["光伏（PV）", "屋顶太阳能板。晴天中午自家发电，从电网买的电就少了。"],
        ["nMAE", "平均绝对误差除以平均用电量，例如 6% 表示平均每小时偏差约 6%。"],
        ["梯度提升树", "很多棵小决策树接力，后一棵专门修正前面的错误，是常用的表格数据预测模型。"],
        ["聚类 / k-means", "不告诉答案，让算法把特征相似的对象自动分成 K 组。"],
        ["训练 / 验证 / 测试集", "分别用来学习、做选择、最终评分；测试集只看一次，防止“偷看答案”。"],
        ["选择偏差", "在评分用的数据上反复挑选方案，导致成绩被高估。"],
        ["Bootstrap 置信区间", "对数据反复重新抽样，看结果的波动范围；区间跨过 0 说明结论不可靠。"],
        ["预测区间 / 分位数", "“有 80% 把握落在这个范围内”；q90 表示 90% 情况下实际值不超过它。"],
        ["校准", "说 80% 的区间，实际也真的有约 80% 命中，说明不确定性估计可信。"],
        ["报童问题 / τ*", "缺货和积压代价不同时的最优订货量规则；τ* 是应当报出的分位数。"],
    ], widths=[4.2, 12])

    path = OUT / "project_report_zh.docx"
    doc.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
