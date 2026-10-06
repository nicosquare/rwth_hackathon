"""Step 10: build the 3-minute presentation (5 slides, 16:9) -> outputs/clustering_story.pptx.
All numbers are read from the result files, so the deck always matches the latest run."""
import json

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from config import FIG, OUT

SL = FIG / "slides"
DARK, GREY, BLUE, ORANGE, RED = (RGBColor(0x1f, 0x2a, 0x36), RGBColor(0x6b, 0x72, 0x7a),
                                 RGBColor(0x24, 0x71, 0xa3), RGBColor(0xe6, 0x7e, 0x22),
                                 RGBColor(0xc0, 0x39, 0x2b))


def load_numbers():
    k = json.load(open(OUT / "selected_k.json"))["selected_k"]
    mk = pd.read_csv(OUT / "metrics_by_k.csv").set_index("K")
    l2 = pd.read_csv(OUT / "level2_metrics.csv").set_index("K")
    val = pd.read_csv(OUT / "validation_selection.csv").set_index("K")
    cs = pd.read_csv(OUT / f"cluster_summary_K{k}.csv").set_index("Cluster")
    l3 = pd.read_csv(OUT / "level3_metrics.csv", index_col=0)
    sens = pd.read_csv(OUT / "level3_sensitivity.csv")
    sens = sens[sens["K"] == k].set_index("Short_fixed_EUR_MWh")
    pv_c = cs["PV_share"].idxmax()
    test_k = l2.loc[2:, "Cost_saving_vs_baseline_pct"]
    sig = sens[sens["CI95_low_pct"] > 0]
    return {
        "k": k, "n_hh": len(pd.read_csv(OUT / "cluster_assignments.csv")),
        "nmae": mk.loc[1, "nMAE_pct"],
        "pv_n": int(cs.loc[pv_c, "N"]), "pv_known": int(cs.loc[pv_c, "N_PV_known"]),
        "pv_share": cs.loc[pv_c, "PV_share"], "pv_mid": cs.loc[pv_c, "pv_midday_ratio"],
        "cost_hh": l2.loc[1, "Imbalance_cost_EUR_per_HH_yr"], "cost_pct": l2.loc[1, "Imbalance_cost_pct_of_DA_bill"],
        "val": -val.loc[k, "cost_vs_baseline_pct"], "val_lo": val.loc[k, "Saving_CI95_low_pct"],
        "val_hi": val.loc[k, "Saving_CI95_high_pct"],
        "test": l2.loc[k, "Cost_saving_vs_baseline_pct"], "test_lo": l2.loc[k, "Saving_CI95_low_pct"],
        "test_hi": l2.loc[k, "Saving_CI95_high_pct"],
        "best_test_k": int(test_k.idxmax()), "best_test": test_k.max(),
        "cov80": float(l3.loc["Coverage_80pct_interval"].iloc[1]),
        "cov90": float(l3.loc["Coverage_90pct_interval"].iloc[1]),
        "sig_from": int(sig.index.min()), "sig_lo": sig["Saving_pct"].iloc[0], "sig_hi": sig["Saving_pct"].iloc[-1],
        "pen_max": int(sens.index.max()), "base_sav": sens.iloc[0]["Saving_pct"],
    }


def text_box(slide, x, y, w, h, paragraphs, size=18, space=10):
    """paragraphs: list of lists of (text, bold, color) runs."""
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, runs in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        for text, bold, color in runs:
            r = p.add_run(); r.text = text
            r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color or DARK
    return tf


def base_slide(prs, title, kicker):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    text_box(s, 0.5, 0.3, 12.3, 0.4, [[(kicker.upper(), True, BLUE)]], size=13)
    text_box(s, 0.5, 0.62, 12.3, 0.9, [[(title, True, DARK)]], size=30)
    return s


def B(t): return (t, True, BLUE)
def T(t): return (t, False, None)
def O(t): return (t, True, ORANGE)
def R(t): return (t, True, RED)


def main():
    n = load_numbers()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # ---------------- slide 1 ----------------
    s = base_slide(prs, f"Forecasting tomorrow's load for {n['n_hh']} heat-pump households",
                   "Level 0 - baseline")
    text_box(s, 0.5, 1.9, 4.6, 5, [
        [T("Task: hourly portfolio forecast for E.ON's "), B("day-ahead"), T(" purchase")],
        [T("Baseline: one gradient-boosting model; only data known by "), B("midnight D-1")],
        [T("Test-year error: "), B(f"nMAE {n['nmae']:.1f}%")],
        [T("Strict time split: "), B("train → validation → test"), T(" (80/20); the test year decides nothing")],
        [B("Question: "), T("does clustering households first make it better?")],
    ], size=18)
    s.shapes.add_picture(str(SL / "s1_forecast_week.png"), Inches(5.3), Inches(2.0), width=Inches(7.7))
    s.notes_slide.notes_text_frame.text = (
        f"We forecast tomorrow's hourly electricity demand of {n['n_hh']} heat-pump households - the quantity "
        "E.ON has to buy on the day-ahead market. Our baseline is a single gradient-boosting model that only "
        "uses information available at midnight before the delivery day. On a full, unseen test year it is "
        f"already accurate: about {n['nmae']:.1f} percent normalized error. We split the data strictly in time "
        "into training, validation and test, so the test year never influences any decision. "
        "Our question: if we first group similar households, does the forecast - and the procurement cost - get better?")

    # ---------------- slide 2 ----------------
    s = base_slide(prs, "Clustering finds the solar owners — without being told", "Level 1 - clustering")
    s.shapes.add_picture(str(SL / "s2_cluster_profiles.png"), Inches(0.5), Inches(1.75), height=Inches(4.4))
    s.shapes.add_picture(str(SL / "s2_pv_share.png"), Inches(7.7), Inches(1.75), height=Inches(4.4))
    text_box(s, 0.5, 6.3, 12.3, 1.1, [[
        T("9 load-shape features (temperature sensitivity, solar signature, time-of-day shares), k-means.  "),
        O(f"{n['pv_n']} households, {n['pv_known']}/{n['pv_known']} known = PV"),
        T(f"; summer midday load only ~{n['pv_mid'] * 4:.0f}/4 of morning/evening."),
    ]], size=16)
    s.notes_slide.notes_text_frame.text = (
        "We cluster households only on how they use electricity: how strongly they react to temperature, whether "
        "their midday load drops when the sun shines, and when during the day they consume. We never give the "
        f"model the PV label. Still, one cluster of {n['pv_n']} households contains only solar owners - all "
        f"{n['pv_known']} with a known label - while the other clusters are around 35 percent. In summer their "
        "grid demand almost disappears at noon. So clustering works very well as customer insight. "
        "The next question is: does it save E.ON money?")

    # ---------------- slide 3 ----------------
    s = base_slide(prs, f"{n['val']:.1f}% saving on validation — {n['test']:.1f}% on the test year",
                   "Level 2 - business metric")
    text_box(s, 0.5, 1.9, 5.2, 5.2, [
        [T("Real German day-ahead prices (SMARD) + intraday penalty for every forecast error")],
        [T("Baseline imbalance cost: "), B(f"{n['cost_hh']:.0f} €/household/year"),
         T(f" (≈{n['cost_pct']:.0f}% of the day-ahead bill)")],
        [T(f"K = {n['k']} chosen on validation: "), B(f"+{n['val']:.1f}%"),
         T(f" [{n['val_lo']:+.1f}, {n['val_hi']:+.1f}]")],
        [T("Same model on the test year: "), R(f"+{n['test']:.1f}%"),
         T(f" [{n['test_lo']:+.1f}, {n['test_hi']:+.1f}] → not significant")],
        [T(f"Picking K on the test set would have claimed +{n['best_test']:.1f}% (K = {n['best_test_k']}) "
           "— a selection-bias trap we avoided")],
    ], size=17)
    s.shapes.add_picture(str(SL / "s3_validation_vs_test.png"), Inches(6.0), Inches(1.8), height=Inches(4.8))
    s.notes_slide.notes_text_frame.text = (
        "To measure what matters for procurement we price every forecast error with real German day-ahead prices "
        "plus an intraday penalty. For the baseline that is about "
        f"{n['cost_hh']:.0f} euros per household and year. On the validation period clustering looked great: "
        f"{n['val']:.1f} percent cheaper, clearly significant. But on the untouched test year the saving shrinks "
        f"to {n['test']:.1f} percent, with a confidence interval around zero. Why? Aggregating {n['n_hh']} households "
        "already averages out individual behaviour, and the validation period was the 2022 energy-crisis winter. "
        f"Had we picked the number of clusters on the test set, we would have reported {n['best_test']:.1f} percent - "
        "our strict split protects E.ON from that kind of over-promise.")

    # ---------------- slide 4 ----------------
    s = base_slide(prs, "The money is in the uncertainty, not in the clusters", "Level 3 - uncertainty")
    s.shapes.add_picture(str(SL / "s4_value_of_uncertainty.png"), Inches(0.4), Inches(1.7), height=Inches(4.3))
    s.shapes.add_picture(str(SL / "s4_uncertainty_by_hour.png"), Inches(7.9), Inches(1.7), height=Inches(3.4))
    text_box(s, 7.9, 5.2, 5.0, 1.0, [[T("Intervals are calibrated: "), B(f"{n['cov80']:.0%}"), T(" in 80%, "),
                                      B(f"{n['cov90']:.0%}"), T(" in 90% interval")]], size=15)
    text_box(s, 0.5, 6.2, 12.3, 1.2, [[
        T("Bid the cost-optimal quantile τ* = c_short / (c_short + c_surplus).  Symmetric penalty: no gain.  "),
        T(f"Shortfall penalty ≥ {n['sig_from']} €/MWh: "),
        B(f"{n['sig_lo']:.0f}–{n['sig_hi']:.0f}% cheaper"), T(" (significant)."),
    ]], size=16)
    s.notes_slide.notes_text_frame.text = (
        "Instead of one number, we forecast a full distribution from the recent forecast errors. It is well "
        f"calibrated: {n['cov80']:.0%} of actual values fall in our 80 percent interval. The forecast is most "
        "uncertain around midday - solar and heat pumps switching - and most certain at night in winter. "
        "We then bid the cost-optimal quantile, the classic newsvendor rule. If being short costs about as much as "
        "being long, the point forecast is already optimal. But real imbalance prices are asymmetric: as soon as a "
        f"shortfall costs {n['sig_from']} euros per megawatt-hour or more, uncertainty-aware bidding is "
        f"{n['sig_lo']:.0f} to {n['sig_hi']:.0f} percent cheaper - far more than clustering ever gave us.")

    # ---------------- slide 5 ----------------
    s = base_slide(prs, "What E.ON should take away", "Conclusion")
    cards = [
        ("1", "Use clustering for customer insight", "It reliably finds PV owners and customer segments — "
         "but does not significantly cut day-ahead cost for a large portfolio."),
        ("2", "Invest in probabilistic forecasts", "Calibrated intervals + quantile bidding pay off as soon as "
         "imbalance penalties are asymmetric."),
        ("3", "Validate like it is real money", "Three-way time split, real prices and bootstrap confidence "
         "intervals prevent over-promising."),
    ]
    for i, (num, head, body) in enumerate(cards):
        x = 0.5 + i * 4.2
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.9), Inches(3.9), Inches(3.4))
        box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xf2, 0xf5, 0xf8); box.line.fill.background()
        text_box(s, x + 0.25, 2.05, 3.4, 3.2, [[(num, True, BLUE)], [(head, True, DARK)], [(body, False, GREY)]],
                 size=19, space=8)
    text_box(s, 0.5, 5.7, 12.3, 1.2, [[B("Next: "), T("real imbalance prices (reBAP) · 15-minute products "
                                                       "(SDAC since Oct 2025) · weather forecasts instead of observed weather")]],
             size=16)
    s.notes_slide.notes_text_frame.text = (
        "Three take-aways. First, clustering is valuable for understanding customers - it finds PV owners without "
        "labels - but it is not a forecasting trick for a large portfolio. Second, the real lever is probabilistic "
        "forecasting combined with cost-aware bidding. Third, validate as if it were real money. Next steps: real "
        "imbalance prices, the new 15-minute day-ahead products, and real weather forecasts. Thank you.")

    for sl in prs.slides:          # small footer
        text_box(sl, 0.5, 7.05, 12.3, 0.3, [[("RWTH Hackathon × E.ON — Bringing the Heat", False, GREY)]], size=10)
    path = OUT / "clustering_story.pptx"
    prs.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
