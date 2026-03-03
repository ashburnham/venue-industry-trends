#!/usr/bin/env python3
"""
Venue Trends Keynote — Data Analysis v2
========================================
Deep analysis of 4,925 competitive analysis reports from Hire Space.
Every insight is cross-tabbed by venue type × event type.
Recommendations are classified by reading all 19,896 texts.

Usage:
    python3 analysis.py
"""

import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Minimum sample size for a cell to be reportable
MIN_N = 10

# Venue type display names
VT_LABELS = {
    "conferenceCentre": "Conference Centre",
    "hotel": "Hotel",
    "historicBuildingsLandmark": "Historic Building",
    "restaurant": "Restaurant",
    "rooftop": "Rooftop",
    "bar": "Bar",
    "meetingRoomsCoWorking": "Meeting Room",
    "outdoor": "Outdoor",
    "theatre": "Theatre",
    "eventVenue": "Event Venue (generic)",
    "gallery": "Gallery",
    "museum": "Museum",
    "cinema": "Cinema",
    "warehouse": "Warehouse",
    "nightclub": "Nightclub",
    "activityVenue": "Activity Venue",
    "activityBar": "Activity Bar",
    "socialGame": "Social/Games",
    "themeVenue": "Theme Venue",
    "gigVenue": "Gig Venue",
    "eventStudio": "Event Studio",
    "boat": "Boat",
}

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#4F46E5",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "bg": "#FFFFFF",
    "text": "#111827",
    "grid": "#E5E7EB",
}
PALETTE = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
           "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
           "#14B8A6", "#A855F7"]

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.5,
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.size": 12,
    "font.family": "sans-serif",
    "legend.fontsize": 11,
    "figure.titlesize": 20,
    "figure.titleweight": "bold",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


def _save(fig, name):
    path = os.path.join(CHART_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [chart] {name}.png")


def _fmt_gbp(x, _=None):
    if abs(x) >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:.0f}"


def _vt_label(vt):
    return VT_LABELS.get(vt, vt)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path="venue-data.json"):
    fpath = os.path.join(BASE_DIR, path)
    print(f"Loading {fpath}...")
    with open(fpath) as f:
        raw = json.load(f)
    print(f"  {len(raw):,} records loaded")
    return raw


def extract_record(r):
    """Flatten a raw record into an analysis-ready dict."""
    tvp = r.get("targetVenueProfile", {})
    m = tvp.get("metrics", {})
    brief = r.get("eventBriefSummarySchema") or {}
    outcome = r.get("outcome") or {}

    quote = m.get("approximateTotalIncTax")
    if not quote or quote <= 0:
        return None

    pp = m.get("pricePositioning") or {}
    or_ = m.get("offeringRichness") or {}

    comps = r.get("competatorProfiles", [])
    comp_quotes = [
        c["metrics"]["approximateTotalIncTax"]
        for c in comps
        if (c.get("metrics") or {}).get("approximateTotalIncTax") and
        c["metrics"]["approximateTotalIncTax"] > 0
    ]

    # Venue type: pick most specific (skip generic 'eventVenue')
    vt_map = r.get("venueTypes") or {}
    active_types = sorted([k for k, v in vt_map.items() if v])
    primary_vt = next((t for t in active_types if t != "eventVenue"), None)
    if primary_vt is None and active_types:
        primary_vt = active_types[0]

    event_date = None
    raw_date = brief.get("eventDate")
    if raw_date:
        try:
            event_date = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    return {
        "quote": quote,
        "margin_pct": m.get("approximateMarginPct"),
        "pp_label": pp.get("label"),
        "pp_value": pp.get("value"),
        "or_value": or_.get("value"),
        "category": r.get("highLevelEventCategory"),
        "venue_type": primary_vt,
        "all_venue_types": active_types,
        "budget": brief.get("budget"),
        "people": brief.get("people"),
        "area": brief.get("perferredArea"),
        "event_date": event_date,
        "probability": outcome.get("expectedProbability"),
        "n_competitors": len(comps),
        "comp_quotes": comp_quotes,
        "comp_mean_quote": statistics.mean(comp_quotes) if comp_quotes else None,
        "recommendations": r.get("recommendations") or [],
        "venue_name": tvp.get("name"),
    }


# ── Recommendation Text Classifier ───────────────────────────────────────────
# Reads every recommendation title+description and classifies into
# concrete, actionable themes based on what the text actually says.

ACTION_THEMES = [
    # (theme_id, display_label, keywords_in_title_or_desc)
    ("bundle_av", "Bundle AV into package/DDR",
     [r'\bav\b', r'\baudio.?visual', r'screen', r'projector', r'mic\b', r'microphone',
      r'sound system', r'tech.?bundle']),
    ("tiered_ddr", "Introduce tiered DDR pricing",
     [r'tiered ddr', r'ddr tier', r'essential.*classic.*premium',
      r'gold.*platinum', r'bronze.*silver.*gold', r'core.*premium.*ddr',
      r'introduce.*ddr']),
    ("tiered_drinks", "Tiered drinks/bar packages",
     [r'tiered.*drink', r'tiered.*bar', r'tiered.*beverage', r'drink.*tier',
      r'bar.*tier', r'drink.*upgrade', r'drink.*upsell', r'open bar',
      r'beer.*wine.*soft', r'beverage.*token', r'bar.*package']),
    ("minimum_spend", "Convert to minimum-spend model",
     [r'minimum.?spend', r'min.?spend', r'f&?b.*minimum', r'food.*bev.*min']),
    ("bundle_festive", "Create festive/Christmas package",
     [r'festive', r'christmas', r'xmas', r'party.*package.*dec',
      r'december.*bundle', r'winter.*wonder', r'holiday.*package']),
    ("bundle_package", "Create all-inclusive event package",
     [r'all.?in', r'all.?inclusive', r'turnkey', r'one.?price',
      r'fixed.?price.*package', r'per.?head.*package', r'package.*pp\b',
      r'bundle.*into.*single', r'transparent.*per.?head']),
    ("upsell_fb", "Upsell F&B / catering add-ons",
     [r'upsell.*canap', r'upsell.*food', r'upsell.*cater', r'upsell.*menu',
      r'pre.?sell.*drink', r'pre.?sell.*bar', r'canapé.*upgrade',
      r'dinner.*upgrade', r'premium.*menu', r'bowl.*food.*upgrade',
      r'street.?food', r'food.*station']),
    ("room_hire_model", "Restructure room-hire pricing",
     [r'room.?hire', r'room.?block', r'room.?credit', r'convert.*hire',
      r'hire.*credit', r'hire.*minimum', r'hire.?rate']),
    ("late_licence", "Late-licence / extended hours add-on",
     [r'late.?licen', r'late.?night', r'midnight', r'extend.*hour',
      r'after.?hour', r'late.?finish']),
    ("proposal_speed", "Faster proposal turnaround",
     [r'proposal.*turnaround', r'turnaround.*proposal', r'response.*time',
      r'quote.*speed', r'hour.*turnaround', r'rapid.*quot',
      r'same.?day.*quot']),
    ("proposal_quality", "Improve proposal presentation/content",
     [r'proposal.*visual', r'proposal.*presentation', r'one.?page',
      r'proposal.*cover', r'case.?stud', r'photo.*testimonial',
      r'floor.?plan', r'imagery', r'render', r'mood.?board',
      r'showcase.*proposal']),
    ("complimentary_add", "Include complimentary extras",
     [r'complimentary', r'free.*welcome', r'free.*arrival',
      r'no.*extra.*cost', r'include.*free', r'gratis',
      r'at no.*charge']),
    ("security_ops", "Bundle security/cloakroom/ops costs",
     [r'security', r'cloakroom', r'door.?staff', r'bouncer',
      r'operations.*cost', r'cleaning.*fee']),
    ("service_charge", "Service charge transparency",
     [r'service.*charge', r'service.*fee', r'gratuity', r'tipping']),
    ("supplier_commission", "Increase supplier commissions/fees",
     [r'commission', r'supplier.*fee', r'caterer.*fee', r'management.*fee',
      r'corkage', r'dry.?hire.*fee']),
    ("external_catering", "External catering licensing/options",
     [r'external.*cater', r'outside.*cater', r'approved.*cater',
      r'caterer.*list', r'catering.*licence', r'dry.?hire']),
    ("capacity_flex", "Address capacity/space flexibility",
     [r'capacity', r'layout.*flex', r'breakout.*room', r'smaller.*room',
      r'room.*option', r'space.*down.?sell', r'alternative.*space',
      r'sub.?\d+.*guest']),
    ("entertainment_exp", "Add entertainment/experience add-ons",
     [r'dj\b', r'photo.?booth', r'magician', r'band\b', r'live.*music',
      r'entertainment', r'game', r'quiz', r'team.?build',
      r'experience.*add', r'immersive']),
    ("seasonal_promo", "Seasonal/off-peak promotions",
     [r'off.?peak', r'mid.?week', r'january.*promot', r'quiet.*period',
      r'seasonal.*rate', r'shoulder.*month', r'summer.*promot',
      r'winter.*promot', r'early.?bird']),
    ("transparency_pricing", "Improve pricing transparency",
     [r'transparent', r'clarity', r'breakdown', r'itemis', r'line.?by.?line',
      r'hidden.*cost', r'no.*surpris', r'vat.*inclusive',
      r'inc.*vat.*headline']),
]

# Compile regexes once
_THEME_PATTERNS = [
    (tid, label, [re.compile(p, re.IGNORECASE) for p in pats])
    for tid, label, pats in ACTION_THEMES
]


def classify_recommendation(title, description):
    """Return list of theme_ids that match this recommendation text."""
    text = f"{title} {description}"
    matched = []
    for tid, _label, patterns in _THEME_PATTERNS:
        if any(p.search(text) for p in patterns):
            matched.append(tid)
    return matched if matched else ["other"]


def _theme_label(tid):
    for t, label, _ in ACTION_THEMES:
        if t == tid:
            return label
    return "Other / Uncategorised"


# ── Module 1: Pricing by Venue Type × Event Type ─────────────────────────────

def module_1_pricing(records):
    print("\n═══ Module 1: Pricing Sweet Spots by Venue Type × Event Type ═══")

    TOP_VTS = _top_venue_types(records, 8)
    TOP_CATS = _top_categories(records, 10)

    # ── 1a: Median quote heatmap — venue type × event type ──
    cross = defaultdict(list)
    for r in records:
        if r["venue_type"] in TOP_VTS and r["category"] in TOP_CATS:
            cross[(r["category"], r["venue_type"])].append(r["quote"])

    _heatmap(cross, TOP_CATS, TOP_VTS,
             title="Median Quote (£): Event Type × Venue Type",
             val_fn=lambda vs: statistics.median(vs),
             fmt_fn=lambda v: _fmt_gbp(v),
             cmap="YlOrRd", vmin=5000, vmax=60000,
             name="1a_quote_heatmap")

    # ── 1b: Budget vs quote by venue type (grouped bar) ──
    vt_data = defaultdict(lambda: {"budgets": [], "quotes": []})
    for r in records:
        if r["venue_type"] and r["budget"] and r["budget"] > 0:
            vt_data[r["venue_type"]]["budgets"].append(r["budget"])
            vt_data[r["venue_type"]]["quotes"].append(r["quote"])

    vts = [vt for vt in TOP_VTS if len(vt_data[vt]["budgets"]) >= MIN_N]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(vts))
    w = 0.35
    med_b = [statistics.median(vt_data[vt]["budgets"]) for vt in vts]
    med_q = [statistics.median(vt_data[vt]["quotes"]) for vt in vts]
    ax.bar(x - w/2, med_b, w, label="Customer Budget", color=COLORS["secondary"],
           edgecolor="white")
    ax.bar(x + w/2, med_q, w, label="Venue Quote", color=COLORS["primary"],
           edgecolor="white")
    for i in range(len(vts)):
        ax.text(x[i] - w/2, med_b[i] + 300, _fmt_gbp(med_b[i]),
                ha="center", fontsize=9, fontweight="bold", color=COLORS["secondary"])
        ax.text(x[i] + w/2, med_q[i] + 300, _fmt_gbp(med_q[i]),
                ha="center", fontsize=9, fontweight="bold", color=COLORS["primary"])
    ax.set_xticks(x)
    ax.set_xticklabels([_vt_label(vt) for vt in vts], rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gbp))
    ax.set_ylabel("Median Amount (£)")
    ax.set_title("Customer Budget vs Venue Quote by Venue Type", pad=15)
    ax.legend()
    fig.tight_layout()
    _save(fig, "1b_budget_vs_quote_by_vt")

    # ── 1c: Budget vs quote by event type ──
    cat_data = defaultdict(lambda: {"budgets": [], "quotes": []})
    for r in records:
        if r["category"] and r["budget"] and r["budget"] > 0:
            cat_data[r["category"]]["budgets"].append(r["budget"])
            cat_data[r["category"]]["quotes"].append(r["quote"])

    cats = [c for c in TOP_CATS if len(cat_data[c]["budgets"]) >= MIN_N]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cats))
    med_b = [statistics.median(cat_data[c]["budgets"]) for c in cats]
    med_q = [statistics.median(cat_data[c]["quotes"]) for c in cats]
    ax.bar(x - w/2, med_b, w, label="Customer Budget", color=COLORS["secondary"],
           edgecolor="white")
    ax.bar(x + w/2, med_q, w, label="Venue Quote", color=COLORS["primary"],
           edgecolor="white")
    for i in range(len(cats)):
        ax.text(x[i] - w/2, med_b[i] + 300, _fmt_gbp(med_b[i]),
                ha="center", fontsize=9, fontweight="bold", color=COLORS["secondary"])
        ax.text(x[i] + w/2, med_q[i] + 300, _fmt_gbp(med_q[i]),
                ha="center", fontsize=9, fontweight="bold", color=COLORS["primary"])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gbp))
    ax.set_ylabel("Median Amount (£)")
    ax.set_title("Customer Budget vs Venue Quote by Event Type", pad=15)
    ax.legend()
    fig.tight_layout()
    _save(fig, "1c_budget_vs_quote_by_cat")

    # ── 1d: Win probability by budget proximity — split by venue type ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes_flat = axes.flatten()
    for idx, vt in enumerate(TOP_VTS[:8]):
        ax = axes_flat[idx]
        vt_recs = [r for r in records if r["venue_type"] == vt
                   and r["budget"] and r["budget"] > 0 and r["probability"] is not None]
        if len(vt_recs) < MIN_N:
            ax.set_visible(False)
            continue
        bins_def = [(0, 0.9, "<90%"), (0.9, 1.1, "90-110%"), (1.1, 1.5, "110-150%"),
                    (1.5, 100, ">150%")]
        labels_b, means_b, counts_b = [], [], []
        for lo, hi, lbl in bins_def:
            vals = [r["probability"] for r in vt_recs if lo <= r["quote"]/r["budget"] < hi]
            if vals:
                labels_b.append(lbl)
                means_b.append(statistics.mean(vals))
                counts_b.append(len(vals))
        if not labels_b:
            ax.set_visible(False)
            continue
        bcols = [COLORS["secondary"], COLORS["primary"], COLORS["accent"],
                 COLORS["danger"]][:len(labels_b)]
        bars = ax.bar(labels_b, means_b, color=bcols, edgecolor="white")
        for bar, v, c in zip(bars, means_b, counts_b):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}%\nn={c}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(_vt_label(vt), fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(means_b) * 1.35)
        ax.set_ylabel("Win Prob (%)" if idx % 4 == 0 else "")
    fig.suptitle("Win Probability by Budget Proximity — By Venue Type",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "1d_budget_proximity_by_vt")

    # Headlines
    for vt in TOP_VTS[:5]:
        d = vt_data[vt]
        if not d["budgets"]:
            continue
        mb = statistics.median(d["budgets"])
        mq = statistics.median(d["quotes"])
        diff = (mq - mb) / mb * 100
        dir_s = "above" if diff > 0 else "below"
        print(f"  {_vt_label(vt)}: budget £{mb:,.0f} → quote £{mq:,.0f} "
              f"({abs(diff):.0f}% {dir_s})")


# ── Module 2: Margins by Venue Type × Event Type ─────────────────────────────

def module_2_margins(records):
    print("\n═══ Module 2: Margin Opportunity Map ═══")

    TOP_VTS = _top_venue_types(records, 10)
    TOP_CATS = _top_categories(records, 10)
    recs = [r for r in records if r["margin_pct"] is not None]

    # ── 2a: Margin heatmap — event type × venue type ──
    cross = defaultdict(list)
    for r in recs:
        if r["venue_type"] in TOP_VTS and r["category"] in TOP_CATS:
            cross[(r["category"], r["venue_type"])].append(r["margin_pct"])

    _heatmap(cross, TOP_CATS, TOP_VTS,
             title="Avg Margin (%): Event Type × Venue Type",
             val_fn=lambda vs: statistics.mean(vs),
             fmt_fn=lambda v: f"{v:.0f}%",
             cmap="RdYlGn", vmin=35, vmax=80,
             name="2a_margin_heatmap")

    # ── 2b: Margin vs win probability by venue type ──
    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, vt in enumerate(TOP_VTS):
        vt_recs = [r for r in recs if r["venue_type"] == vt and r["probability"] is not None]
        if len(vt_recs) < MIN_N:
            continue
        avg_m = statistics.mean(r["margin_pct"] for r in vt_recs)
        avg_p = statistics.mean(r["probability"] for r in vt_recs)
        ax.scatter(avg_m, avg_p, s=len(vt_recs) * 0.8, c=PALETTE[idx % len(PALETTE)],
                   alpha=0.7, edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(_vt_label(vt), (avg_m, avg_p),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.set_xlabel("Average Margin (%)")
    ax.set_ylabel("Average Win Probability (%)")
    ax.set_title("Margin vs Win Rate: The Venue Type Trade-off", pad=15)
    fig.tight_layout()
    _save(fig, "2b_margin_vs_win_by_vt")

    # ── 2c: Margin distribution by venue type (box plot) ──
    vt_margins = defaultdict(list)
    for r in recs:
        if r["venue_type"] in TOP_VTS:
            vt_margins[r["venue_type"]].append(r["margin_pct"])
    vts_plot = [vt for vt in TOP_VTS if len(vt_margins[vt]) >= MIN_N]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot([vt_margins[vt] for vt in vts_plot],
                    tick_labels=[_vt_label(vt) for vt in vts_plot],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([_vt_label(vt) for vt in vts_plot], rotation=30, ha="right")
    ax.set_ylabel("Estimated Margin (%)")
    ax.set_title("Margin Distribution by Venue Type", pad=15)
    fig.tight_layout()
    _save(fig, "2c_margin_dist_by_vt")

    # ── 2d: Margin by event type (horizontal bar) ──
    cat_margins = defaultdict(list)
    for r in recs:
        if r["category"]:
            cat_margins[r["category"]].append(r["margin_pct"])
    cats = sorted([c for c in cat_margins if len(cat_margins[c]) >= MIN_N],
                  key=lambda c: -statistics.mean(cat_margins[c]))

    fig, ax = plt.subplots(figsize=(14, 7))
    means = [statistics.mean(cat_margins[c]) for c in cats]
    colors = [COLORS["primary"] if m >= 65 else COLORS["accent"] if m >= 55
              else COLORS["danger"] for m in means]
    bars = ax.barh(cats[::-1], means[::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, means[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Average Margin (%)")
    ax.set_title("Average Margin by Event Type", pad=15)
    ax.set_xlim(0, max(means) * 1.12)
    fig.tight_layout()
    _save(fig, "2d_margin_by_category")

    # Headlines
    all_margins = [r["margin_pct"] for r in recs]
    print(f"  Overall: median {statistics.median(all_margins):.0f}%, "
          f"mean {statistics.mean(all_margins):.0f}%")
    for vt in TOP_VTS[:5]:
        ms = vt_margins.get(vt, [])
        if ms:
            vt_recs_p = [r for r in recs if r["venue_type"] == vt and r["probability"] is not None]
            wp = statistics.mean(r["probability"] for r in vt_recs_p) if vt_recs_p else 0
            print(f"  {_vt_label(vt)}: {statistics.mean(ms):.0f}% margin, "
                  f"{wp:.0f}% win prob (n={len(ms)})")


# ── Module 3: Price Positioning vs Win Probability ────────────────────────────

def module_3_price_win(records):
    print("\n═══ Module 3: Price vs Win Probability ═══")

    TOP_VTS = _top_venue_types(records, 8)
    TOP_CATS = _top_categories(records, 6)
    recs = [r for r in records if r["pp_value"] is not None and r["probability"] is not None]

    # ── 3a: Overall curve ──
    pp_bins = defaultdict(list)
    for r in recs:
        pp_bins[r["pp_value"]].append(r["probability"])

    fig, ax = plt.subplots(figsize=(12, 7))
    pvs = sorted(pp_bins.keys())
    mps = [statistics.mean(pp_bins[v]) for v in pvs]
    cnts = [len(pp_bins[v]) for v in pvs]
    sizes = [max(50, min(400, c * 0.5)) for c in cnts]
    ax.scatter(pvs, mps, s=sizes, c=COLORS["primary"], alpha=0.7,
               edgecolors="white", linewidth=1.5, zorder=5)
    ax.plot(pvs, mps, color=COLORS["primary"], linewidth=2, alpha=0.5)
    for v, p, c in zip(pvs, mps, cnts):
        ax.annotate(f"{p:.1f}%\n(n={c})", (v, p), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Price Positioning (1=Budget → 10=Premium)")
    ax.set_ylabel("Mean Win Probability (%)")
    ax.set_title("Win Probability vs Price Positioning", pad=15)
    ax.set_xticks(range(1, 11))
    fig.tight_layout()
    _save(fig, "3a_price_vs_win")

    # ── 3b: By venue type (small multiples) ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for idx, vt in enumerate(TOP_VTS[:8]):
        ax = axes[idx // 4][idx % 4]
        vt_pp = defaultdict(list)
        for r in recs:
            if r["venue_type"] == vt:
                vt_pp[r["pp_value"]].append(r["probability"])
        pvs = sorted(vt_pp.keys())
        if not pvs:
            ax.set_visible(False)
            continue
        mps = [statistics.mean(vt_pp[v]) for v in pvs]
        cnts = [len(vt_pp[v]) for v in pvs]
        ax.bar(pvs, mps, color=COLORS["primary"], edgecolor="white", alpha=0.8)
        for v, p, c in zip(pvs, mps, cnts):
            if c >= 5:
                ax.text(v, p + 0.5, f"{p:.0f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(_vt_label(vt), fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 65)
        ax.set_ylabel("Win %" if idx % 4 == 0 else "")
    fig.suptitle("Price Positioning vs Win Probability — By Venue Type",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "3b_price_vs_win_by_vt")

    # ── 3c: By event type (small multiples) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for idx, cat in enumerate(TOP_CATS[:6]):
        ax = axes[idx // 3][idx % 3]
        cat_pp = defaultdict(list)
        for r in recs:
            if r["category"] == cat:
                cat_pp[r["pp_value"]].append(r["probability"])
        pvs = sorted(cat_pp.keys())
        mps = [statistics.mean(cat_pp[v]) for v in pvs]
        cnts = [len(cat_pp[v]) for v in pvs]
        ax.bar(pvs, mps, color=COLORS["primary"], edgecolor="white", alpha=0.8)
        for v, p, c in zip(pvs, mps, cnts):
            if c >= 5:
                ax.text(v, p + 0.5, f"{p:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(cat, fontsize=13, fontweight="bold")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 65)
    fig.suptitle("Price Positioning vs Win Probability — By Event Type",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "3c_price_vs_win_by_cat")

    # ── 3d: Value quadrant coloured by win prob ──
    recs_q = [r for r in records if r["pp_value"] and r["or_value"] and r["probability"]]
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter([r["pp_value"] for r in recs_q],
                         [r["or_value"] for r in recs_q],
                         c=[r["probability"] for r in recs_q],
                         cmap="RdYlGn", s=15, alpha=0.4, vmin=10, vmax=70)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Win Probability (%)")
    ax.axhline(5.5, color=COLORS["neutral"], linestyle="--", alpha=0.4)
    ax.axvline(5.5, color=COLORS["neutral"], linestyle="--", alpha=0.4)
    ax.text(2, 9, "VALUE\nLEADERS", ha="center", fontsize=14, fontweight="bold",
            color=COLORS["secondary"], alpha=0.6)
    ax.text(9, 9, "PREMIUM\nJUSTIFIED", ha="center", fontsize=14, fontweight="bold",
            color=COLORS["primary"], alpha=0.6)
    ax.text(2, 2, "BASIC &\nCHEAP", ha="center", fontsize=14, fontweight="bold",
            color=COLORS["neutral"], alpha=0.6)
    ax.text(9, 2, "OVER-\nPRICED", ha="center", fontsize=14, fontweight="bold",
            color=COLORS["danger"], alpha=0.6)
    ax.set_xlabel("Price Positioning (1=Budget → 10=Premium)")
    ax.set_ylabel("Offering Richness (1=Basic → 10=Premium)")
    ax.set_title("Value Quadrant: Price vs Offering Quality", pad=15)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    fig.tight_layout()
    _save(fig, "3d_value_quadrant")

    # Headlines — pricing cliffs
    print("  Pricing cliffs (>5pp drop):")
    for vt in TOP_VTS[:5]:
        vt_pp = defaultdict(list)
        for r in recs:
            if r["venue_type"] == vt:
                vt_pp[r["pp_value"]].append(r["probability"])
        points = [(v, statistics.mean(vt_pp[v]), len(vt_pp[v]))
                  for v in sorted(vt_pp.keys()) if len(vt_pp[v]) >= MIN_N]
        for i in range(len(points) - 1):
            drop = points[i][1] - points[i+1][1]
            if drop > 5:
                print(f"    {_vt_label(vt)}: {points[i][1]:.0f}% → {points[i+1][1]:.0f}% "
                      f"(−{drop:.0f}pp) at position {points[i][0]}→{points[i+1][0]}")


# ── Module 4: Competitive Landscape by Venue Type ────────────────────────────

def module_4_competitive(records):
    print("\n═══ Module 4: Competitive Landscape ═══")

    TOP_VTS = _top_venue_types(records, 10)
    TOP_CATS = _top_categories(records, 10)
    recs = [r for r in records if r["comp_quotes"]]

    # ── 4a: % most expensive / cheapest heatmap by venue type × event type ──
    cross_exp = defaultdict(lambda: {"most": 0, "cheap": 0, "total": 0})
    for r in recs:
        if r["venue_type"] in TOP_VTS and r["category"] in TOP_CATS:
            key = (r["category"], r["venue_type"])
            cross_exp[key]["total"] += 1
            if r["quote"] >= max(r["comp_quotes"]):
                cross_exp[key]["most"] += 1
            if r["quote"] <= min(r["comp_quotes"]):
                cross_exp[key]["cheap"] += 1

    _heatmap(cross_exp, TOP_CATS, TOP_VTS,
             title="% of Proposals Where Venue Is Most Expensive",
             val_fn=lambda d: d["most"] / d["total"] * 100 if d["total"] >= MIN_N else None,
             fmt_fn=lambda v: f"{v:.0f}%",
             cmap="RdYlGn_r", vmin=5, vmax=50,
             name="4a_most_expensive_heatmap",
             raw_dict=True)

    # ── 4b: Price gap vs competitors by venue type ──
    vt_gaps = defaultdict(list)
    for r in recs:
        if r["venue_type"] and r["comp_mean_quote"] and r["comp_mean_quote"] > 0:
            gap = (r["quote"] - r["comp_mean_quote"]) / r["comp_mean_quote"] * 100
            vt_gaps[r["venue_type"]].append(gap)

    vts = [vt for vt in TOP_VTS if len(vt_gaps[vt]) >= MIN_N]
    fig, ax = plt.subplots(figsize=(14, 7))
    med_gaps = [statistics.median(vt_gaps[vt]) for vt in vts]
    colors = [COLORS["danger"] if g > 10 else COLORS["accent"] if g > 0
              else COLORS["secondary"] for g in med_gaps]
    bars = ax.barh([_vt_label(vt) for vt in vts][::-1], med_gaps[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.axvline(0, color=COLORS["neutral"], linewidth=2, alpha=0.5)
    for bar, val in zip(bars, med_gaps[::-1]):
        ax.text(bar.get_width() + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f"{val:+.0f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Median Price Gap vs Competitors (%)")
    ax.set_title("How Much More/Less Do Venue Types Charge vs Competitors?", pad=15)
    fig.tight_layout()
    _save(fig, "4b_price_gap_by_vt")

    # ── 4c: Being cheapest vs most expensive → win prob, by venue type ──
    vt_pos = defaultdict(lambda: {"cheap_p": [], "mid_p": [], "exp_p": []})
    for r in recs:
        if r["probability"] is None or not r["venue_type"]:
            continue
        if r["quote"] <= min(r["comp_quotes"]):
            vt_pos[r["venue_type"]]["cheap_p"].append(r["probability"])
        elif r["quote"] >= max(r["comp_quotes"]):
            vt_pos[r["venue_type"]]["exp_p"].append(r["probability"])
        else:
            vt_pos[r["venue_type"]]["mid_p"].append(r["probability"])

    fig, ax = plt.subplots(figsize=(14, 8))
    vts_p = [vt for vt in TOP_VTS if len(vt_pos[vt]["cheap_p"]) >= 5
             and len(vt_pos[vt]["exp_p"]) >= 5]
    x = np.arange(len(vts_p))
    w = 0.25
    cheap_m = [statistics.mean(vt_pos[vt]["cheap_p"]) for vt in vts_p]
    mid_m = [statistics.mean(vt_pos[vt]["mid_p"]) if vt_pos[vt]["mid_p"]
             else 0 for vt in vts_p]
    exp_m = [statistics.mean(vt_pos[vt]["exp_p"]) for vt in vts_p]
    ax.bar(x - w, cheap_m, w, label="Cheapest", color=COLORS["secondary"], edgecolor="white")
    ax.bar(x, mid_m, w, label="Mid-range", color=COLORS["accent"], edgecolor="white")
    ax.bar(x + w, exp_m, w, label="Most Expensive", color=COLORS["danger"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_vt_label(vt) for vt in vts_p], rotation=30, ha="right")
    ax.set_ylabel("Mean Win Probability (%)")
    ax.set_title("Win Probability by Price Position — By Venue Type", pad=15)
    ax.legend()
    fig.tight_layout()
    _save(fig, "4c_price_position_win_by_vt")

    # Headlines
    for vt in TOP_VTS[:5]:
        d = vt_pos[vt]
        if d["cheap_p"] and d["exp_p"]:
            c_avg = statistics.mean(d["cheap_p"])
            e_avg = statistics.mean(d["exp_p"])
            print(f"  {_vt_label(vt)}: cheapest wins {c_avg:.0f}% vs "
                  f"most expensive {e_avg:.0f}% ({c_avg-e_avg:+.0f}pp)")


# ── Module 5: Deep Recommendation Analysis ───────────────────────────────────

def module_5_recommendations(records):
    print("\n═══ Module 5: Deep Recommendation Analysis (19,896 texts) ═══")

    TOP_VTS = _top_venue_types(records, 10)
    TOP_CATS = _top_categories(records, 10)

    # Classify every recommendation
    all_classified = []  # (theme_id, category, venue_type, annual_impact, title)
    theme_counts = Counter()
    theme_by_vt = defaultdict(lambda: Counter())
    theme_by_cat = defaultdict(lambda: Counter())
    theme_impacts = defaultdict(list)
    theme_examples = defaultdict(list)

    total_recs = 0
    for r in records:
        for rec in r.get("recommendations", []):
            total_recs += 1
            title = rec.get("title", "")
            desc = rec.get("description", "")
            themes = classify_recommendation(title, desc)
            ai = rec.get("annualImpact")
            annual = ai["value"] if isinstance(ai, dict) and ai.get("value") is not None else None

            for tid in themes:
                theme_counts[tid] += 1
                if r["venue_type"]:
                    theme_by_vt[r["venue_type"]][tid] += 1
                if r["category"]:
                    theme_by_cat[r["category"]][tid] += 1
                if annual is not None and annual > 0:
                    theme_impacts[tid].append(annual)
                if len(theme_examples[tid]) < 5:
                    theme_examples[tid].append(title)

                all_classified.append((tid, r["category"], r["venue_type"], annual, title))

    print(f"  Classified {total_recs:,} recommendations into {len(theme_counts)} themes")

    # ── 5a: Top themes overall (bar chart) ──
    top_themes = theme_counts.most_common(20)
    fig, ax = plt.subplots(figsize=(16, 9))
    labels = [_theme_label(tid) for tid, _ in top_themes]
    values = [cnt for _, cnt in top_themes]
    bars = ax.barh(labels[::-1], values[::-1], color=COLORS["primary"], edgecolor="white")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Recommendations")
    ax.set_title("What Are Venues Actually Being Told To Do? (All 19,896 Recommendations)",
                 pad=15)
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()
    _save(fig, "5a_recommendation_themes")

    # ── 5b: Theme heatmap — venue type × theme ──
    top_theme_ids = [tid for tid, _ in top_themes[:12]]
    top_theme_labels = [_theme_label(tid) for tid in top_theme_ids]

    fig, ax = plt.subplots(figsize=(18, 10))
    hm = np.zeros((len(TOP_VTS), len(top_theme_ids)))
    for i, vt in enumerate(TOP_VTS):
        total_vt = sum(theme_by_vt[vt].values())
        for j, tid in enumerate(top_theme_ids):
            if total_vt > 0:
                hm[i, j] = theme_by_vt[vt].get(tid, 0) / total_vt * 100
    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_theme_ids)))
    ax.set_xticklabels(top_theme_labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(TOP_VTS)))
    ax.set_yticklabels([_vt_label(vt) for vt in TOP_VTS], fontsize=11)
    for i in range(len(TOP_VTS)):
        for j in range(len(top_theme_ids)):
            v = hm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > 15 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of venue type's recommendations")
    ax.set_title("Recommendation Themes by Venue Type (% of all recs for that type)", pad=15)
    fig.tight_layout()
    _save(fig, "5b_themes_by_venue_type")

    # ── 5c: Theme heatmap — event type × theme ──
    fig, ax = plt.subplots(figsize=(18, 10))
    hm2 = np.zeros((len(TOP_CATS), len(top_theme_ids)))
    for i, cat in enumerate(TOP_CATS):
        total_cat = sum(theme_by_cat[cat].values())
        for j, tid in enumerate(top_theme_ids):
            if total_cat > 0:
                hm2[i, j] = theme_by_cat[cat].get(tid, 0) / total_cat * 100
    im2 = ax.imshow(hm2, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_theme_ids)))
    ax.set_xticklabels(top_theme_labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(TOP_CATS)))
    ax.set_yticklabels(TOP_CATS, fontsize=11)
    for i in range(len(TOP_CATS)):
        for j in range(len(top_theme_ids)):
            v = hm2[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > 15 else "black")
    cbar2 = fig.colorbar(im2, ax=ax, shrink=0.8)
    cbar2.set_label("% of event type's recommendations")
    ax.set_title("Recommendation Themes by Event Type (% of all recs for that type)", pad=15)
    fig.tight_layout()
    _save(fig, "5c_themes_by_event_type")

    # ── 5d: Annual revenue impact by theme ──
    themes_with_impact = [(tid, theme_impacts[tid])
                          for tid in [t for t, _ in top_themes[:15]]
                          if len(theme_impacts[tid]) >= 5]
    fig, ax = plt.subplots(figsize=(16, 8))
    t_labels = [_theme_label(tid) for tid, _ in themes_with_impact]
    t_medians = [statistics.median(vals) for _, vals in themes_with_impact]
    t_counts = [len(vals) for _, vals in themes_with_impact]
    # Sort by median impact
    sorted_idx = sorted(range(len(t_medians)), key=lambda i: t_medians[i])
    bars = ax.barh([t_labels[i] for i in sorted_idx],
                   [t_medians[i] for i in sorted_idx],
                   color=COLORS["secondary"], edgecolor="white")
    for bar, idx_s in zip(bars, sorted_idx):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"£{t_medians[idx_s]:,.0f} (n={t_counts[idx_s]})",
                va="center", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gbp))
    ax.set_xlabel("Median Estimated Annual Revenue Impact (£)")
    ax.set_title("Revenue Impact by Recommendation Theme", pad=15)
    fig.tight_layout()
    _save(fig, "5d_impact_by_theme")

    # ── 5e: Top specific recommendations by venue type (text table chart) ──
    # For the top 5 venue types, find their #1 theme and show examples
    print("\n  Top recommendation theme per venue type:")
    for vt in TOP_VTS[:8]:
        if not theme_by_vt[vt]:
            continue
        top_tid, top_cnt = theme_by_vt[vt].most_common(1)[0]
        total = sum(theme_by_vt[vt].values())
        pct = top_cnt / total * 100
        print(f"    {_vt_label(vt)}: \"{_theme_label(top_tid)}\" "
              f"({top_cnt} recs, {pct:.0f}% of all)")
        # Show example titles
        examples = [t for tid, cat, vtype, ai, t in all_classified
                    if tid == top_tid and vtype == vt][:2]
        for ex in examples:
            print(f"      e.g. \"{ex}\"")

    print("\n  Top recommendation theme per event type:")
    for cat in TOP_CATS[:8]:
        if not theme_by_cat[cat]:
            continue
        top_tid, top_cnt = theme_by_cat[cat].most_common(1)[0]
        total = sum(theme_by_cat[cat].values())
        pct = top_cnt / total * 100
        print(f"    {cat}: \"{_theme_label(top_tid)}\" "
              f"({top_cnt} recs, {pct:.0f}% of all)")
        examples = [t for tid, c, vtype, ai, t in all_classified
                    if tid == top_tid and c == cat][:2]
        for ex in examples:
            print(f"      e.g. \"{ex}\"")

    # Revenue impact headlines
    print("\n  Highest-impact themes (median annual £):")
    for tid, vals in sorted(themes_with_impact, key=lambda x: -statistics.median(x[1]))[:5]:
        print(f"    {_theme_label(tid)}: £{statistics.median(vals):,.0f}/yr "
              f"(n={len(vals)})")

    return all_classified


# ── Module 6: Demand & Seasonality by Venue Type ─────────────────────────────

def module_6_seasonal(records):
    print("\n═══ Module 6: Demand & Seasonal Trends ═══")

    TOP_VTS = _top_venue_types(records, 6)
    TOP_CATS = _top_categories(records, 6)
    recs = [r for r in records if r["event_date"] is not None]
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    months = list(range(1, 13))

    # ── 6a: Demand heatmap — venue type × month ──
    vt_month = defaultdict(lambda: Counter())
    for r in recs:
        if r["venue_type"]:
            vt_month[r["venue_type"]][r["event_date"].month] += 1

    fig, ax = plt.subplots(figsize=(16, 8))
    hm = np.zeros((len(TOP_VTS), 12))
    for i, vt in enumerate(TOP_VTS):
        for j, m in enumerate(months):
            hm[i, j] = vt_month[vt].get(m, 0)
    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels([month_names[m] for m in months])
    ax.set_yticks(range(len(TOP_VTS)))
    ax.set_yticklabels([_vt_label(vt) for vt in TOP_VTS])
    for i in range(len(TOP_VTS)):
        for j in range(12):
            v = int(hm[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if v > 80 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Number of Proposals")
    ax.set_title("Demand by Venue Type × Month", pad=15)
    fig.tight_layout()
    _save(fig, "6a_demand_vt_month")

    # ── 6b: Demand by event type × month (stacked) ──
    cat_month = defaultdict(lambda: Counter())
    for r in recs:
        if r["category"]:
            cat_month[r["category"]][r["event_date"].month] += 1

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(12)
    for idx, cat in enumerate(TOP_CATS):
        vals = [cat_month[cat].get(m, 0) for m in months]
        ax.bar([month_names[m] for m in months], vals, bottom=bottom,
               label=cat, color=PALETTE[idx], edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)
    ax.set_ylabel("Number of Proposals")
    ax.set_title("Event Type Demand by Month", pad=15)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    _save(fig, "6b_event_type_by_month")

    # ── 6c: Median budget by venue type × month ──
    vt_month_budget = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["venue_type"] and r["budget"] and r["budget"] > 0:
            vt_month_budget[r["venue_type"]][r["event_date"].month].append(r["budget"])

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, vt in enumerate(TOP_VTS[:5]):
        meds = []
        ms_plot = []
        for m in months:
            vals = vt_month_budget[vt].get(m, [])
            if len(vals) >= 5:
                meds.append(statistics.median(vals))
                ms_plot.append(m)
        if ms_plot:
            ax.plot(ms_plot, meds, marker="o", label=_vt_label(vt),
                    color=PALETTE[idx], linewidth=2, markersize=6)
    ax.set_xticks(months)
    ax.set_xticklabels([month_names[m] for m in months])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gbp))
    ax.set_ylabel("Median Budget (£)")
    ax.set_title("Median Budget by Month — Top Venue Types", pad=15)
    ax.legend()
    fig.tight_layout()
    _save(fig, "6c_budget_by_month_vt")

    # ── 6d: Guest count by venue type (box plot) ──
    vt_people = defaultdict(list)
    for r in records:
        if r["venue_type"] and r["people"] and r["people"] > 0:
            vt_people[r["venue_type"]].append(r["people"])

    vts = [vt for vt in TOP_VTS if len(vt_people[vt]) >= MIN_N]
    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot([vt_people[vt] for vt in vts],
                    tick_labels=[_vt_label(vt) for vt in vts],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([_vt_label(vt) for vt in vts], rotation=30, ha="right")
    ax.set_ylabel("Guest Count")
    ax.set_title("Guest Count Distribution by Venue Type", pad=15)
    fig.tight_layout()
    _save(fig, "6d_guests_by_vt")

    # Headlines
    for vt in TOP_VTS[:4]:
        peak_m = max(months, key=lambda m: vt_month[vt].get(m, 0))
        cnt = vt_month[vt].get(peak_m, 0)
        ppl = vt_people.get(vt, [])
        med_ppl = statistics.median(ppl) if ppl else 0
        print(f"  {_vt_label(vt)}: peak {month_names[peak_m]} ({cnt} proposals), "
              f"median {med_ppl:.0f} guests")


# ── Module 7: Budget-Reality Gap by Venue Type × Event Type ──────────────────

def module_7_budget_gap(records):
    print("\n═══ Module 7: The Budget-Reality Gap ═══")

    TOP_VTS = _top_venue_types(records, 10)
    TOP_CATS = _top_categories(records, 10)
    recs = [r for r in records if r["budget"] and r["budget"] > 0]

    # ── 7a: Quote/budget ratio heatmap — venue type × event type ──
    cross = defaultdict(list)
    for r in recs:
        if r["venue_type"] in TOP_VTS and r["category"] in TOP_CATS:
            cross[(r["category"], r["venue_type"])].append(r["quote"] / r["budget"])

    _heatmap(cross, TOP_CATS, TOP_VTS,
             title="Median Quote/Budget Ratio: Event Type × Venue Type",
             val_fn=lambda vs: statistics.median(vs),
             fmt_fn=lambda v: f"{v:.2f}x",
             cmap="RdYlGn_r", vmin=0.7, vmax=1.6,
             name="7a_budget_ratio_heatmap")

    # ── 7b: Quote/budget ratio by venue type ──
    vt_ratios = defaultdict(list)
    for r in recs:
        if r["venue_type"]:
            vt_ratios[r["venue_type"]].append(r["quote"] / r["budget"])

    vts = [vt for vt in TOP_VTS if len(vt_ratios[vt]) >= MIN_N]
    fig, ax = plt.subplots(figsize=(14, 7))
    meds = [statistics.median(vt_ratios[vt]) for vt in vts]
    sorted_idx = sorted(range(len(meds)), key=lambda i: meds[i], reverse=True)
    colors = [COLORS["danger"] if meds[i] > 1.15 else COLORS["accent"] if meds[i] > 1.0
              else COLORS["secondary"] for i in sorted_idx]
    bars = ax.barh([_vt_label(vts[i]) for i in sorted_idx],
                   [meds[i] for i in sorted_idx],
                   color=colors, edgecolor="white")
    ax.axvline(1.0, color=COLORS["neutral"], linewidth=2, alpha=0.5, linestyle="--",
               label="Budget = Quote")
    for bar, i in zip(bars, sorted_idx):
        pct = (meds[i] - 1) * 100
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{meds[i]:.2f}x ({pct:+.0f}%)", va="center", fontsize=11,
                fontweight="bold")
    ax.set_xlabel("Median Quote / Budget Ratio")
    ax.set_title("Which Venue Types Overquote Most?", pad=15)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "7b_overquote_by_vt")

    # ── 7c: Win probability by budget proximity, by event type ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    top6cats = [c for c in TOP_CATS[:6]]
    for idx, cat in enumerate(top6cats):
        ax = axes[idx // 3][idx % 3]
        cat_recs = [r for r in recs if r["category"] == cat and r["probability"] is not None]
        bins_def = [(0, 0.9, "<90%"), (0.9, 1.1, "90-110%"),
                    (1.1, 1.5, "110-150%"), (1.5, 100, ">150%")]
        ls, ms, cs = [], [], []
        for lo, hi, lbl in bins_def:
            vals = [r["probability"] for r in cat_recs if lo <= r["quote"]/r["budget"] < hi]
            if vals:
                ls.append(lbl)
                ms.append(statistics.mean(vals))
                cs.append(len(vals))
        bcols = [COLORS["secondary"], COLORS["primary"], COLORS["accent"],
                 COLORS["danger"]][:len(ls)]
        bars = ax.bar(ls, ms, color=bcols, edgecolor="white")
        for bar, v, c in zip(bars, ms, cs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}%\nn={c}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(cat, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(ms) * 1.35 if ms else 50)
        ax.set_ylabel("Win Prob (%)" if idx % 3 == 0 else "")
    fig.suptitle("Win Probability by Budget Proximity — By Event Type",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "7c_budget_prox_by_cat")

    # ── 7d: Overquote curve — gap vs win prob ──
    bins_data = defaultdict(list)
    for r in recs:
        if r["probability"] is None:
            continue
        gap = (r["quote"] / r["budget"] - 1) * 100
        b = int(gap // 20) * 20
        b = max(-60, min(b, 200))
        bins_data[b].append(r["probability"])

    fig, ax = plt.subplots(figsize=(12, 7))
    sb = sorted(bins_data.keys())
    bm = [statistics.mean(bins_data[b]) for b in sb]
    bc = [len(bins_data[b]) for b in sb]
    sizes = [max(30, min(300, c * 0.3)) for c in bc]
    colors = [COLORS["secondary"] if b <= 0 else COLORS["accent"] if b <= 40
              else COLORS["danger"] for b in sb]
    ax.scatter(sb, bm, s=sizes, c=colors, alpha=0.8, edgecolors="white",
               linewidth=1.5, zorder=5)
    ax.plot(sb, bm, color=COLORS["neutral"], linewidth=1.5, alpha=0.5)
    for b, p, c in zip(sb, bm, bc):
        if c >= 30:
            ax.annotate(f"{p:.0f}%", (b, p), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
    ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Quote vs Budget Gap (%)")
    ax.set_ylabel("Mean Win Probability (%)")
    ax.set_title("Win Probability Falls as Overquoting Increases", pad=15)
    fig.tight_layout()
    _save(fig, "7d_overquote_curve")

    # Headlines
    all_ratios = [r["quote"] / r["budget"] for r in recs]
    over = sum(1 for ratio in all_ratios if ratio > 1.0)
    print(f"  Overall: {over/len(all_ratios)*100:.0f}% of proposals exceed budget")
    for vt in TOP_VTS[:5]:
        rs = vt_ratios.get(vt, [])
        if rs:
            med = statistics.median(rs)
            print(f"  {_vt_label(vt)}: median {med:.2f}x budget "
                  f"({(med-1)*100:+.0f}%)")


# ── Helper functions ──────────────────────────────────────────────────────────

def _top_venue_types(records, n):
    """Return top n venue types by frequency (excluding tiny types)."""
    vt_counts = Counter(r["venue_type"] for r in records if r["venue_type"])
    return [vt for vt, _ in vt_counts.most_common(n)]


def _top_categories(records, n):
    """Return top n event categories by frequency."""
    cat_counts = Counter(r["category"] for r in records if r["category"])
    return [c for c, _ in cat_counts.most_common(n)]


def _heatmap(cross, rows, cols, title, val_fn, fmt_fn, cmap, vmin, vmax, name,
             raw_dict=False):
    """Generic heatmap builder for row × col cross-tabs."""
    fig, ax = plt.subplots(figsize=(16, 9))
    hm = np.full((len(rows), len(cols)), np.nan)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            data = cross.get((row, col))
            if data is None:
                continue
            if raw_dict:
                val = val_fn(data)
            else:
                if isinstance(data, list) and len(data) >= MIN_N:
                    val = val_fn(data)
                else:
                    val = None
            if val is not None:
                hm[i, j] = val

    im = ax.imshow(hm, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_vt_label(c) for c in cols], rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=11)
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = hm[i, j]
            if not np.isnan(val):
                # Determine text colour based on position in range
                norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                text_color = "white" if norm > 0.65 or norm < 0.2 else "black"
                ax.text(j, i, fmt_fn(val), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, pad=15)
    fig.tight_layout()
    _save(fig, name)


# ── Summary Report ────────────────────────────────────────────────────────────

def print_summary(records):
    print("\n" + "=" * 72)
    print("  VENUE TRENDS KEYNOTE — SUMMARY REPORT")
    print("=" * 72)

    total = len(records)
    with_cat = sum(1 for r in records if r["category"])
    with_budget = sum(1 for r in records if r["budget"] and r["budget"] > 0)
    with_prob = sum(1 for r in records if r["probability"] is not None)

    print(f"\n  Dataset: {total:,} proposals analysed")
    print(f"  With event category: {with_cat:,} ({with_cat/total*100:.0f}%)")
    print(f"  With customer budget: {with_budget:,} ({with_budget/total*100:.0f}%)")
    print(f"  With win probability: {with_prob:,} ({with_prob/total*100:.0f}%)")

    quotes = [r["quote"] for r in records]
    margins = [r["margin_pct"] for r in records if r["margin_pct"] is not None]
    probs = [r["probability"] for r in records if r["probability"] is not None]

    print(f"\n  Median quote: £{statistics.median(quotes):,.0f}")
    print(f"  Median margin: {statistics.median(margins):.0f}%")
    print(f"  Mean win probability: {statistics.mean(probs):.1f}%")

    print(f"\n  Top venue types:")
    for vt, cnt in Counter(r["venue_type"] for r in records if r["venue_type"]).most_common(8):
        print(f"    {_vt_label(vt)}: {cnt:,}")

    print(f"\n  Top event categories:")
    for cat, cnt in Counter(r["category"] for r in records if r["category"]).most_common(8):
        print(f"    {cat}: {cnt:,}")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw = load_data()

    print("Extracting records...")
    records = [extract_record(r) for r in raw]
    records = [r for r in records if r is not None]
    print(f"  {len(records):,} usable records")

    module_1_pricing(records)
    module_2_margins(records)
    module_3_price_win(records)
    module_4_competitive(records)
    module_5_recommendations(records)
    module_6_seasonal(records)
    module_7_budget_gap(records)

    print_summary(records)
    print("\nDone. All charts in ./charts/")


if __name__ == "__main__":
    main()
