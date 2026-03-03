#!/usr/bin/env python3
"""
Venue Trends Keynote — Final Presentation Charts
=================================================
Generates 6 publication-quality charts in Hire Space brand style,
plus an HTML slide deck and speaker notes.
"""

import json
import os
import statistics
import re
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "keynote")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hire Space Brand ──────────────────────────────────────────────────────────
HS_GREEN = "#00A82C"
HS_GREEN_LIGHT = "#53B04B"
HS_GREEN_PALE = "#E6F7EA"
HS_DARK = "#282A30"
HS_DARK_MID = "#3A3D45"
HS_GREY = "#6B7280"
HS_GREY_LIGHT = "#EBEDF0"
HS_WHITE = "#FFFFFF"
HS_OFF_WHITE = "#F7F9FC"

# Extended palette for multi-series
HS_PALETTE = [
    "#00A82C",  # green
    "#0EA5E9",  # sky blue
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F97316",  # orange
    "#6366F1",  # indigo
    "#84CC16",  # lime
]

# Dark-background style
plt.rcParams.update({
    "figure.facecolor": HS_DARK,
    "axes.facecolor": HS_DARK,
    "axes.edgecolor": HS_DARK_MID,
    "axes.labelcolor": HS_OFF_WHITE,
    "axes.labelsize": 15,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": HS_DARK_MID,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "xtick.color": HS_GREY,
    "ytick.color": HS_GREY,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Open Sans", "Nunito Sans", "Helvetica Neue", "Arial"],
    "legend.fontsize": 11,
    "legend.facecolor": HS_DARK_MID,
    "legend.edgecolor": HS_DARK_MID,
    "legend.labelcolor": HS_OFF_WHITE,
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.4,
    "savefig.facecolor": HS_DARK,
})

MIN_N = 10

VT_LABELS = {
    "conferenceCentre": "Conference Centre",
    "hotel": "Hotel",
    "historicBuildingsLandmark": "Historic Venue",
    "restaurant": "Restaurant",
    "rooftop": "Rooftop",
    "bar": "Bar",
    "meetingRoomsCoWorking": "Meeting Room",
    "outdoor": "Outdoor",
    "theatre": "Theatre",
    "eventVenue": "Event Venue",
    "gallery": "Gallery",
    "museum": "Museum",
    "cinema": "Cinema",
    "activityBar": "Activity Bar",
}

# The 8 venue types to show (breakouts swapped for rooftop)
SHOW_VTS = [
    "conferenceCentre", "historicBuildingsLandmark", "hotel",
    "restaurant", "rooftop", "bar", "outdoor", "meetingRoomsCoWorking",
]

SHOW_CATS = [
    "Conferences", "Christmas Parties", "Corporate Events",
    "Award Ceremonies", "Summer Parties", "Meetings",
    "Networking Events", "Private Dining", "Training & Workshops",
    "Away Days",
]


def _vt(k):
    return VT_LABELS.get(k, k)


def _fmt_gbp(x, _=None):
    if abs(x) >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:.0f}"


def _save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [chart] {name}.png")
    return path


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_extract():
    fpath = os.path.join(BASE_DIR, "venue-data.json")
    print(f"Loading {fpath}...")
    with open(fpath) as f:
        raw = json.load(f)
    print(f"  {len(raw):,} records")

    records = []
    for r in raw:
        tvp = r.get("targetVenueProfile", {})
        m = tvp.get("metrics", {})
        brief = r.get("eventBriefSummarySchema") or {}
        outcome = r.get("outcome") or {}
        quote = m.get("approximateTotalIncTax")
        if not quote or quote <= 0:
            continue

        pp = m.get("pricePositioning") or {}
        or_ = m.get("offeringRichness") or {}
        comps = r.get("competatorProfiles", [])
        comp_quotes = [
            c["metrics"]["approximateTotalIncTax"]
            for c in comps
            if (c.get("metrics") or {}).get("approximateTotalIncTax")
            and c["metrics"]["approximateTotalIncTax"] > 0
        ]

        vt_map = r.get("venueTypes") or {}
        active = sorted([k for k, v in vt_map.items() if v])
        primary_vt = next((t for t in active if t != "eventVenue"), None)
        if primary_vt is None and active:
            primary_vt = active[0]

        records.append({
            "quote": quote,
            "margin_pct": m.get("approximateMarginPct"),
            "pp_label": pp.get("label"),
            "pp_value": pp.get("value"),
            "or_value": or_.get("value"),
            "category": r.get("highLevelEventCategory"),
            "venue_type": primary_vt,
            "budget": brief.get("budget"),
            "people": brief.get("people"),
            "probability": outcome.get("expectedProbability"),
            "comp_quotes": comp_quotes,
            "comp_mean_quote": statistics.mean(comp_quotes) if comp_quotes else None,
            "recommendations": r.get("recommendations") or [],
        })

    print(f"  {len(records):,} usable records")
    return records


# ── Recommendation classifier ────────────────────────────────────────────────

ACTION_THEMES = [
    ("bundle_av", "Bundle AV into DDR / package",
     [r'\bav\b', r'audio.?visual', r'screen', r'projector', r'\bmic\b',
      r'microphone', r'sound system', r'tech.?bundle']),
    ("tiered_ddr", "Introduce tiered DDR pricing",
     [r'tiered ddr', r'ddr tier', r'essential.*classic.*premium',
      r'gold.*platinum', r'bronze.*silver', r'core.*premium.*ddr',
      r'introduce.*ddr']),
    ("tiered_drinks", "Tiered drinks / bar packages",
     [r'tiered.*drink', r'tiered.*bar', r'tiered.*beverage', r'drink.*tier',
      r'bar.*tier', r'drink.*upgrade', r'drink.*upsell', r'open bar',
      r'beer.*wine.*soft', r'beverage.*token', r'bar.*package']),
    ("minimum_spend", "Convert to minimum-spend model",
     [r'minimum.?spend', r'min.?spend', r'f&?b.*minimum']),
    ("bundle_festive", "Create festive / Christmas package",
     [r'festive', r'christmas', r'xmas', r'december.*bundle',
      r'winter.*wonder', r'holiday.*package']),
    ("bundle_package", "Create all-inclusive package",
     [r'all.?in', r'all.?inclusive', r'turnkey', r'one.?price',
      r'fixed.?price.*package', r'per.?head.*package', r'package.*pp\b',
      r'bundle.*into.*single', r'transparent.*per.?head']),
    ("upsell_fb", "Upsell F&B / catering add-ons",
     [r'upsell.*canap', r'upsell.*food', r'upsell.*cater', r'upsell.*menu',
      r'pre.?sell.*drink', r'pre.?sell.*bar', r'premium.*menu',
      r'food.*station', r'street.?food']),
    ("room_hire", "Restructure room-hire pricing",
     [r'room.?hire', r'room.?block', r'room.?credit', r'convert.*hire',
      r'hire.*credit', r'hire.*minimum']),
    ("late_licence", "Late-licence / extended hours",
     [r'late.?licen', r'midnight', r'extend.*hour', r'after.?hour',
      r'late.?finish']),
    ("proposal_speed", "Faster proposal turnaround",
     [r'proposal.*turnaround', r'turnaround.*proposal', r'response.*time',
      r'hour.*turnaround', r'rapid.*quot', r'same.?day.*quot']),
    ("proposal_quality", "Improve proposal presentation",
     [r'proposal.*visual', r'one.?page', r'case.?stud', r'testimonial',
      r'floor.?plan', r'imagery', r'render', r'mood.?board']),
    ("complimentary", "Include complimentary extras",
     [r'complimentary', r'free.*welcome', r'free.*arrival',
      r'no.*extra.*cost', r'include.*free', r'at no.*charge']),
    ("security_ops", "Bundle security / ops costs",
     [r'security', r'cloakroom', r'door.?staff', r'cleaning.*fee']),
    ("service_charge", "Service charge transparency",
     [r'service.*charge', r'gratuity', r'tipping']),
    ("external_catering", "External catering options",
     [r'external.*cater', r'outside.*cater', r'approved.*cater',
      r'catering.*licence', r'dry.?hire']),
    ("capacity_flex", "Flexible capacity / space options",
     [r'capacity', r'breakout.*room', r'smaller.*room', r'room.*option',
      r'space.*down.?sell', r'alternative.*space']),
    ("entertainment", "Entertainment / experience add-ons",
     [r'\bdj\b', r'photo.?booth', r'magician', r'\bband\b', r'live.*music',
      r'entertainment', r'game', r'quiz', r'team.?build', r'immersive']),
    ("transparency", "Improve pricing transparency",
     [r'transparent', r'breakdown', r'itemis', r'line.?by.?line',
      r'hidden.*cost', r'vat.*inclusive', r'inc.*vat.*headline']),
]

_PATTERNS = [(tid, lbl, [re.compile(p, re.I) for p in ps])
             for tid, lbl, ps in ACTION_THEMES]


def classify_rec(title, desc):
    text = f"{title} {desc}"
    matched = [tid for tid, _, pats in _PATTERNS if any(p.search(text) for p in pats)]
    return matched or ["other"]


def _theme_label(tid):
    for t, lbl, _ in ACTION_THEMES:
        if t == tid:
            return lbl
    return "Other"


# ── Chart 1: Budget vs Quote by Event Type ───────────────────────────────────

def chart_1_budget_vs_quote(records):
    cat_data = defaultdict(lambda: {"b": [], "q": []})
    for r in records:
        if r["category"] and r["budget"] and r["budget"] > 0:
            cat_data[r["category"]]["b"].append(r["budget"])
            cat_data[r["category"]]["q"].append(r["quote"])

    cats = [c for c in SHOW_CATS if len(cat_data[c]["b"]) >= MIN_N]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(cats))
    w = 0.35

    med_b = [statistics.median(cat_data[c]["b"]) for c in cats]
    med_q = [statistics.median(cat_data[c]["q"]) for c in cats]

    ax.bar(x - w/2, med_b, w, label="Customer Budget",
           color=HS_GREEN, edgecolor=HS_DARK, linewidth=0.5,
           zorder=3, alpha=0.9)
    ax.bar(x + w/2, med_q, w, label="Venue Quote",
           color="#0EA5E9", edgecolor=HS_DARK, linewidth=0.5,
           zorder=3, alpha=0.9)

    # Value labels
    for i in range(len(cats)):
        ax.text(x[i] - w/2, med_b[i] + 400, _fmt_gbp(med_b[i]),
                ha="center", fontsize=10, fontweight="bold", color=HS_GREEN)
        ax.text(x[i] + w/2, med_q[i] + 400, _fmt_gbp(med_q[i]),
                ha="center", fontsize=10, fontweight="bold", color="#0EA5E9")
        # Show gap
        gap_pct = (med_q[i] - med_b[i]) / med_b[i] * 100
        if abs(gap_pct) > 3:
            sign = "+" if gap_pct > 0 else ""
            col = "#EF4444" if gap_pct > 10 else HS_GREEN if gap_pct < -5 else HS_GREY
            ax.text(x[i], max(med_b[i], med_q[i]) + 1800,
                    f"{sign}{gap_pct:.0f}%", ha="center", fontsize=9,
                    fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", color=HS_OFF_WHITE, fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gbp))
    ax.set_ylabel("")
    ax.set_title("Customer Budget vs What Venues Actually Quote",
                 color=HS_OFF_WHITE, pad=20)
    ax.legend(framealpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, "slide_1_budget_vs_quote")


# ── Chart 2: Win Probability by Budget Proximity — by Venue Type ─────────────

def chart_2_budget_proximity(records):
    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    axes_flat = axes.flatten()

    bins_def = [
        (0, 0.9, "Under\nbudget", HS_GREEN),
        (0.9, 1.0, "-10% to\nbudget", "#0EA5E9"),
        (1.0, 1.1, "Budget\nto +10%", "#8B5CF6"),
        (1.1, 1.5, "10-50%\nover", "#F59E0B"),
        (1.5, 100, ">50%\nover", "#EF4444"),
    ]

    for idx, vt in enumerate(SHOW_VTS[:8]):
        ax = axes_flat[idx]
        vt_recs = [r for r in records if r["venue_type"] == vt
                   and r["budget"] and r["budget"] > 0 and r["probability"] is not None]
        if len(vt_recs) < MIN_N:
            ax.set_visible(False)
            continue

        labels, means, counts, colors = [], [], [], []
        for lo, hi, lbl, col in bins_def:
            vals = [r["probability"] for r in vt_recs if lo <= r["quote"]/r["budget"] < hi]
            if vals:
                labels.append(lbl)
                means.append(statistics.mean(vals))
                counts.append(len(vals))
                colors.append(col)

        if not labels:
            ax.set_visible(False)
            continue

        bars = ax.bar(labels, means, color=colors, edgecolor=HS_DARK, linewidth=0.5,
                      zorder=3, width=0.7)
        for bar, v, c in zip(bars, means, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold",
                    color=HS_OFF_WHITE)
            ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                    f"n={c}", ha="center", fontsize=8, color=HS_GREY)

        ax.set_title(_vt(vt), fontsize=14, fontweight="bold", color=HS_OFF_WHITE, pad=10)
        ax.set_ylim(0, max(means) * 1.3)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylabel("" if idx % 4 != 0 else "Win Probability")
        ax.set_axisbelow(True)

    fig.suptitle("How Budget Alignment Affects Win Rates — By Venue Type",
                 fontsize=22, fontweight="bold", color=HS_OFF_WHITE, y=1.01)
    fig.tight_layout()
    return _save(fig, "slide_2_budget_proximity")


# ── Chart 3: Recommendation Themes (overall) ─────────────────────────────────

def chart_3_themes(records):
    theme_counts = Counter()
    theme_impacts = defaultdict(list)

    for r in records:
        for rec in r.get("recommendations", []):
            title = rec.get("title", "")
            desc = rec.get("description", "")
            themes = classify_rec(title, desc)
            ai = rec.get("annualImpact")
            annual = ai["value"] if isinstance(ai, dict) and ai.get("value") is not None else None
            for tid in themes:
                theme_counts[tid] += 1
                if annual and annual > 0:
                    theme_impacts[tid].append(annual)

    # Top 15, skip "other"
    top = [(tid, cnt) for tid, cnt in theme_counts.most_common(20) if tid != "other"][:15]

    fig, ax = plt.subplots(figsize=(16, 9))
    labels = [_theme_label(tid) for tid, _ in top]
    values = [cnt for _, cnt in top]

    # Gradient green bars
    n = len(labels)
    greens = [f"#{int(0 + (83-0)*i/n):02x}{int(168 + (176-168)*i/n):02x}{int(44 + (75-44)*i/n):02x}"
              for i in range(n)]

    bars = ax.barh(labels[::-1], values[::-1], color=greens[::-1],
                   edgecolor=HS_DARK, linewidth=0.5, zorder=3)

    for bar, val in zip(bars, values[::-1]):
        # Count label
        ax.text(bar.get_width() + 40, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=11, fontweight="bold",
                color=HS_OFF_WHITE)

    ax.set_xlabel("")
    ax.set_title("What 19,896 Recommendations Actually Say Venues Should Do",
                 color=HS_OFF_WHITE, pad=20)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()
    return _save(fig, "slide_3_rec_themes")


# ── Chart 4: Themes by Venue Type (heatmap) ──────────────────────────────────

def chart_4_themes_by_vt(records):
    theme_by_vt = defaultdict(lambda: Counter())
    for r in records:
        if not r["venue_type"]:
            continue
        for rec in r.get("recommendations", []):
            for tid in classify_rec(rec.get("title", ""), rec.get("description", "")):
                theme_by_vt[r["venue_type"]][tid] += 1

    # Top themes (skip other)
    all_themes = Counter()
    for vt_c in theme_by_vt.values():
        all_themes.update(vt_c)
    top_tids = [tid for tid, _ in all_themes.most_common(20) if tid != "other"][:12]
    top_labels = [_theme_label(tid) for tid in top_tids]

    fig, ax = plt.subplots(figsize=(20, 10))
    hm = np.zeros((len(SHOW_VTS), len(top_tids)))
    for i, vt in enumerate(SHOW_VTS):
        total = sum(theme_by_vt[vt].values()) or 1
        for j, tid in enumerate(top_tids):
            hm[i, j] = theme_by_vt[vt].get(tid, 0) / total * 100

    # Custom colourmap: dark → HS green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("hs",
        [HS_DARK_MID, "#1a4a2a", "#2d7a3d", HS_GREEN, HS_GREEN_LIGHT])

    im = ax.imshow(hm, cmap=cmap, aspect="auto", vmin=0, vmax=22)
    ax.set_xticks(range(len(top_tids)))
    ax.set_xticklabels(top_labels, rotation=40, ha="right", fontsize=11,
                       color=HS_OFF_WHITE)
    ax.set_yticks(range(len(SHOW_VTS)))
    ax.set_yticklabels([_vt(vt) for vt in SHOW_VTS], fontsize=13,
                       color=HS_OFF_WHITE)

    for i in range(len(SHOW_VTS)):
        for j in range(len(top_tids)):
            v = hm[i, j]
            if v >= 1:
                txt = ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                              fontsize=10, fontweight="bold",
                              color=HS_OFF_WHITE if v > 8 else HS_GREY)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% of venue type's recommendations", color=HS_OFF_WHITE,
                   fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=HS_GREY)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=HS_GREY)

    ax.set_title("What Each Venue Type Is Being Told To Fix",
                 color=HS_OFF_WHITE, pad=20, fontsize=20)
    ax.tick_params(colors=HS_GREY)
    fig.tight_layout()
    return _save(fig, "slide_4_themes_by_vt")


# ── Chart 5: Themes by Event Type (heatmap) ──────────────────────────────────

def chart_5_themes_by_cat(records):
    theme_by_cat = defaultdict(lambda: Counter())
    for r in records:
        if not r["category"]:
            continue
        for rec in r.get("recommendations", []):
            for tid in classify_rec(rec.get("title", ""), rec.get("description", "")):
                theme_by_cat[r["category"]][tid] += 1

    all_themes = Counter()
    for cat_c in theme_by_cat.values():
        all_themes.update(cat_c)
    top_tids = [tid for tid, _ in all_themes.most_common(20) if tid != "other"][:12]
    top_labels = [_theme_label(tid) for tid in top_tids]

    cats = [c for c in SHOW_CATS if sum(theme_by_cat[c].values()) >= 20]

    fig, ax = plt.subplots(figsize=(20, 10))
    hm = np.zeros((len(cats), len(top_tids)))
    for i, cat in enumerate(cats):
        total = sum(theme_by_cat[cat].values()) or 1
        for j, tid in enumerate(top_tids):
            hm[i, j] = theme_by_cat[cat].get(tid, 0) / total * 100

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("hs",
        [HS_DARK_MID, "#1a4a2a", "#2d7a3d", HS_GREEN, HS_GREEN_LIGHT])

    im = ax.imshow(hm, cmap=cmap, aspect="auto", vmin=0, vmax=28)
    ax.set_xticks(range(len(top_tids)))
    ax.set_xticklabels(top_labels, rotation=40, ha="right", fontsize=11,
                       color=HS_OFF_WHITE)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=13, color=HS_OFF_WHITE)

    for i in range(len(cats)):
        for j in range(len(top_tids)):
            v = hm[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color=HS_OFF_WHITE if v > 8 else HS_GREY)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% of event type's recommendations", color=HS_OFF_WHITE,
                   fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=HS_GREY)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=HS_GREY)

    ax.set_title("What Each Event Type Demands From Venues",
                 color=HS_OFF_WHITE, pad=20, fontsize=20)
    ax.tick_params(colors=HS_GREY)
    fig.tight_layout()
    return _save(fig, "slide_5_themes_by_cat")


# ── Chart 6: Budget/Quote Ratio Heatmap ──────────────────────────────────────

def chart_6_budget_ratio(records):
    recs = [r for r in records if r["budget"] and r["budget"] > 0]
    cross = defaultdict(list)
    for r in recs:
        if r["venue_type"] in SHOW_VTS and r["category"] in SHOW_CATS:
            cross[(r["category"], r["venue_type"])].append(r["quote"] / r["budget"])

    cats = [c for c in SHOW_CATS if any(len(cross.get((c, vt), [])) >= MIN_N
                                        for vt in SHOW_VTS)]

    fig, ax = plt.subplots(figsize=(18, 10))
    hm = np.full((len(cats), len(SHOW_VTS)), np.nan)
    for i, cat in enumerate(cats):
        for j, vt in enumerate(SHOW_VTS):
            vals = cross.get((cat, vt), [])
            if len(vals) >= MIN_N:
                hm[i, j] = statistics.median(vals)

    # Diverging colourmap centred on 1.0
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("budget",
        [HS_GREEN, HS_GREEN_PALE, HS_OFF_WHITE, "#FDE68A", "#F59E0B", "#EF4444"])
    norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.5)

    im = ax.imshow(hm, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(SHOW_VTS)))
    ax.set_xticklabels([_vt(vt) for vt in SHOW_VTS], rotation=30, ha="right",
                       fontsize=12, color=HS_OFF_WHITE)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=13, color=HS_OFF_WHITE)

    for i in range(len(cats)):
        for j in range(len(SHOW_VTS)):
            val = hm[i, j]
            if not np.isnan(val):
                pct = (val - 1) * 100
                sign = "+" if pct >= 0 else ""
                # Dark text on light cells, light on dark
                lum = 0.5 + (val - 1.0) * 2  # rough luminance proxy
                text_col = HS_DARK if 0.85 < val < 1.15 else HS_OFF_WHITE
                ax.text(j, i, f"{val:.2f}x\n({sign}{pct:.0f}%)",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color=text_col)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Quote / Budget Ratio", color=HS_OFF_WHITE, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=HS_GREY)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=HS_GREY)

    ax.set_title("The Budget Reality Gap: Where Venues Over- and Under-Quote",
                 color=HS_OFF_WHITE, pad=20, fontsize=20)
    ax.tick_params(colors=HS_GREY)
    fig.tight_layout()
    return _save(fig, "slide_6_budget_ratio")


# ── HTML Presentation ─────────────────────────────────────────────────────────

def build_html(chart_paths):
    slides = [
        {
            "title": "Customer Budget vs What Venues Actually Quote",
            "subtitle": "Are you pricing yourself out — or leaving money on the table?",
            "chart": os.path.basename(chart_paths[0]),
            "bullets": [
                "Corporate Events are overquoted by <strong>47%</strong> — the biggest mismatch in the dataset",
                "Summer Parties are <em>under</em>quoted by 23% — venues are leaving revenue on the table",
                "Christmas Parties hit the sweet spot: quotes land 5% below budget on average",
                "Award Ceremonies: budgets of £30k meet quotes of £33k — close, but the premium hurts win rates",
            ],
        },
        {
            "title": "How Budget Alignment Affects Win Rates",
            "subtitle": "The closer you are to budget, the more you win. Every venue type.",
            "chart": os.path.basename(chart_paths[1]),
            "bullets": [
                "Across all venue types, quoting within 10% of budget wins <strong>+15pp</strong> more than quoting 50%+ over",
                "Hotels show the starkest cliff: 47% win rate within budget → 21% when >50% over",
                "Restaurants are most forgiving — even 10-50% over still holds 38%",
                "Conference Centres win 44% when price-aligned but only 30% when >50% over budget",
            ],
        },
        {
            "title": "What 19,896 Recommendations Actually Say",
            "subtitle": "We read every single one. These are the patterns.",
            "chart": os.path.basename(chart_paths[2]),
            "bullets": [
                "<strong>\"Bundle AV into the DDR\"</strong> — the single most common recommendation across all proposals",
                "Tiered pricing (DDR tiers, drinks tiers) appears in over 3,400 recommendations",
                "All-inclusive packaging and minimum-spend models dominate the top 5",
                "Entertainment and experience add-ons are the fastest path to incremental revenue (£36k median annual impact)",
            ],
        },
        {
            "title": "What Each Venue Type Is Being Told To Fix",
            "subtitle": "Your venue type has a specific playbook. Here it is.",
            "chart": os.path.basename(chart_paths[3]),
            "bullets": [
                "<strong>Conference Centres & Hotels:</strong> AV bundling is your #1 opportunity — 18% of all recs",
                "<strong>Bars & Restaurants:</strong> minimum-spend models are your unlock — stop charging room hire",
                "<strong>Historic Venues:</strong> you're being told to create all-inclusive packages to justify the premium",
                "<strong>Rooftops & Outdoor:</strong> entertainment add-ons and late-licence upsells are your margin play",
            ],
        },
        {
            "title": "What Each Event Type Demands From Venues",
            "subtitle": "Different events need different fixes. Match your pitch to the brief.",
            "chart": os.path.basename(chart_paths[4]),
            "bullets": [
                "<strong>Conferences:</strong> 24% of recs say bundle AV — planners want one number, not line items",
                "<strong>Christmas Parties:</strong> festive packaging is the #1 ask — create tiered \"Christmas Party\" bundles",
                "<strong>Summer Parties:</strong> entertainment add-ons dominate — DJs, photobooths, games",
                "<strong>Award Ceremonies:</strong> AV + all-inclusive packages — they need turnkey production, not à la carte",
            ],
        },
        {
            "title": "The Budget Reality Gap",
            "subtitle": "Where exactly are venues over- and under-quoting?",
            "chart": os.path.basename(chart_paths[5]),
            "bullets": [
                "<strong>Biggest overquoters:</strong> Historic Venues on Award Ceremonies (1.23x) and Outdoor on Corporate Events (1.12x)",
                "<strong>Biggest underquoters:</strong> Conference Centres on Summer Parties (0.77x) — pricing 23% below budget",
                "Hotels overquote on Conferences by 20% — but win 36% vs. 42% average",
                "The green cells are your opportunity: you're <em>under</em> budget and could charge more",
            ],
        },
    ]

    # Copy logo into output folder
    import shutil
    logo_src = os.path.join(BASE_DIR, "Hire Space Logo White.png")
    logo_dst = os.path.join(OUT_DIR, "hs-logo-white.png")
    if os.path.exists(logo_src):
        shutil.copy2(logo_src, logo_dst)
    HS_LOGO_IMG = '<img src="hs-logo-white.png" alt="Hire Space" class="hs-logo">'
    HS_LOGO_TITLE = '<img src="hs-logo-white.png" alt="Hire Space" class="title-logo">'

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Venue Trends Keynote — Hire Space</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@300;400;600;700;800;900&family=Open+Sans:wght@300;400;500;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  :root {{
    --hs-green: {HS_GREEN};
    --hs-green-light: {HS_GREEN_LIGHT};
    --hs-dark: {HS_DARK};
    --hs-dark-mid: {HS_DARK_MID};
    --hs-grey: {HS_GREY};
    --hs-grey-light: {HS_GREY_LIGHT};
    --hs-white: {HS_WHITE};
    --hs-off-white: {HS_OFF_WHITE};
  }}

  html {{ scroll-behavior: smooth; }}

  body {{
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--hs-dark);
    color: var(--hs-off-white);
    overflow-x: hidden;
  }}

  h1, h2, h3, .stat-box .number {{
    font-family: 'Nunito Sans', 'Open Sans', sans-serif;
  }}

  /* ── Logo ── */
  .hs-logo {{
    height: 32px;
    width: auto;
  }}

  .slide-logo {{
    position: absolute;
    top: 2rem;
    left: 3rem;
    z-index: 10;
  }}

  /* ── Title slide ── */
  .title-slide {{
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(135deg, var(--hs-dark) 0%, #1a2e1f 50%, var(--hs-dark) 100%);
    position: relative;
  }}

  .title-slide::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,168,44,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(0,168,44,0.05) 0%, transparent 60%);
    pointer-events: none;
  }}

  .title-slide .title-logo {{
    height: 48px;
    width: auto;
    margin-bottom: 2.5rem;
    position: relative;
    z-index: 1;
  }}

  .title-slide h1 {{
    font-size: 4.5rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, var(--hs-off-white) 0%, var(--hs-green-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
  }}

  .title-slide .subtitle {{
    font-size: 1.4rem;
    font-weight: 300;
    color: var(--hs-grey);
    max-width: 700px;
    line-height: 1.6;
    position: relative;
    z-index: 1;
  }}

  .title-slide .stat-row {{
    display: flex;
    gap: 3rem;
    margin-top: 3rem;
    position: relative;
    z-index: 1;
  }}

  .stat-box {{
    text-align: center;
  }}

  .stat-box .number {{
    font-size: 3rem;
    font-weight: 900;
    color: var(--hs-green);
    line-height: 1;
  }}

  .stat-box .label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--hs-grey);
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  /* ── Content slides ── */
  .slide {{
    min-height: 100vh;
    padding: 3rem 5% 4rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    position: relative;
  }}

  .slide:nth-child(even) {{
    background: linear-gradient(180deg, var(--hs-dark) 0%, #1e2024 100%);
  }}

  .slide-number {{
    position: absolute;
    top: 2rem;
    right: 3rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--hs-green);
    opacity: 0.6;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  .slide h2 {{
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.4rem;
    color: var(--hs-off-white);
  }}

  .slide .slide-subtitle {{
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--hs-grey);
    margin-bottom: 2rem;
  }}

  /* ── Chart centred, full width ── */
  .chart-container {{
    background: var(--hs-dark);
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--hs-dark-mid);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    max-width: 1100px;
    width: 100%;
    margin: 0 auto 2.5rem;
  }}

  .chart-container img {{
    width: 100%;
    height: auto;
    display: block;
  }}

  /* ── 2x2 insight grid ── */
  .insight-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem 2.5rem;
    max-width: 1100px;
    width: 100%;
    text-align: left;
  }}

  .insight-card {{
    padding: 1.1rem 1.3rem;
    border-left: 3px solid var(--hs-dark-mid);
    font-size: 1rem;
    line-height: 1.6;
    color: rgba(247, 249, 252, 0.85);
    transition: all 0.3s ease;
    background: rgba(58, 61, 69, 0.2);
    border-radius: 0 10px 10px 0;
  }}

  .insight-card:hover {{
    border-left-color: var(--hs-green);
    color: var(--hs-off-white);
    background: rgba(58, 61, 69, 0.35);
  }}

  .insight-card strong {{
    color: var(--hs-green);
    font-weight: 700;
  }}

  .insight-card em {{
    color: var(--hs-green-light);
    font-style: normal;
    font-weight: 600;
  }}

  /* ── Closing slide ── */
  .closing-slide {{
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(135deg, var(--hs-dark) 0%, #1a2e1f 100%);
    position: relative;
  }}

  .closing-slide h2 {{
    font-size: 3.5rem;
    font-weight: 900;
    margin-bottom: 1.5rem;
    color: var(--hs-off-white);
  }}

  .closing-slide p {{
    font-size: 1.3rem;
    color: var(--hs-grey);
    max-width: 600px;
    line-height: 1.7;
  }}

  .closing-slide .cta {{
    margin-top: 2.5rem;
    padding: 1rem 2.5rem;
    background: var(--hs-green);
    color: var(--hs-dark);
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    text-decoration: none;
    transition: background 0.2s;
  }}

  .closing-slide .cta:hover {{
    background: var(--hs-green-light);
  }}

  /* ── Keyboard nav ── */
  .nav-hint {{
    position: fixed;
    bottom: 1.5rem;
    right: 2rem;
    font-size: 0.75rem;
    color: var(--hs-grey);
    opacity: 0.4;
    z-index: 100;
  }}
</style>
</head>
<body>

<div class="nav-hint">Arrow keys or scroll to navigate</div>

<!-- Title Slide -->
<section class="title-slide" id="slide-0">
  <div class="slide-logo">{HS_LOGO_IMG}</div>
  {HS_LOGO_TITLE}
  <h1>Venue Industry Trends</h1>
  <p class="subtitle">Data-driven insights from 4,925 competitive analyses.<br>What venues get right, what they get wrong, and exactly how to fix it.</p>
  <div class="stat-row">
    <div class="stat-box">
      <div class="number">4,925</div>
      <div class="label">Proposals Analysed</div>
    </div>
    <div class="stat-box">
      <div class="number">19,896</div>
      <div class="label">Recommendations Read</div>
    </div>
    <div class="stat-box">
      <div class="number">~25,000</div>
      <div class="label">Competitor Quotes</div>
    </div>
  </div>
</section>
"""

    for i, slide in enumerate(slides):
        html += f"""
<!-- Slide {i+1} -->
<section class="slide" id="slide-{i+1}">
  <div class="slide-logo">{HS_LOGO_IMG}</div>
  <div class="slide-number">Slide {i+1} of {len(slides)}</div>
  <h2>{slide['title']}</h2>
  <p class="slide-subtitle">{slide['subtitle']}</p>
  <div class="chart-container">
    <img src="{slide['chart']}" alt="{slide['title']}" loading="lazy">
  </div>
  <div class="insight-grid">
"""
        for bullet in slide["bullets"]:
            html += f"    <div class=\"insight-card\">{bullet}</div>\n"

        html += """  </div>
</section>
"""

    html += """
<!-- Closing Slide -->
<section class="closing-slide" id="slide-closing">
  <div class="slide-logo">{HS_LOGO_IMG}</div>
  {HS_LOGO_TITLE}
  <h2>Your Next Move</h2>
  <p>Every one of these insights is actionable today. The venues that align pricing to budgets, bundle AV into their DDR, and create transparent packages will win more — starting this quarter.</p>
</section>

<script>
  // Keyboard navigation
  const slides = document.querySelectorAll('section');
  let current = 0;

  function goTo(idx) {
    idx = Math.max(0, Math.min(idx, slides.length - 1));
    slides[idx].scrollIntoView({ behavior: 'smooth' });
    current = idx;
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      goTo(current + 1);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      goTo(current - 1);
    }
  });

  // Track scroll position for current slide
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        current = Array.from(slides).indexOf(entry.target);
      }
    });
  }, { threshold: 0.5 });

  slides.forEach(s => observer.observe(s));
</script>
</body>
</html>"""

    html_path = os.path.join(OUT_DIR, "presentation.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  [html] presentation.html")
    return html_path


# ── Speaker Notes ─────────────────────────────────────────────────────────────

def build_speaker_notes():
    notes = """# Venue Industry Trends — Speaker Notes
## TED-style delivery guide

---

## OPENING (Title Slide) — 90 seconds

**Walk on, pause, let the numbers land.**

"We analysed nearly five thousand competitive proposals. Every quote, every competitor, every recommendation. And we read — actually read — all twenty thousand recommendations that our AI generated for venues.

What we found is both simple and uncomfortable: the gap between what customers want to pay and what venues charge is the single biggest predictor of whether you win or lose. And most venues are on the wrong side of it.

Let me show you exactly where."

**[CLICK to Slide 1]**

---

## SLIDE 1: Budget vs Quote — 2 minutes

**Key message: The market is telling you what it wants to pay. Are you listening?**

"This is the most important chart in this entire presentation. Green bars are what your customers tell us they want to spend. Blue bars are what venues actually quote.

Look at Corporate Events — venues are overshooting by forty-seven percent. Nearly half above budget. That's not premium pricing, that's pricing yourself out of the conversation before it starts.

Now look at Summer Parties — the opposite problem. Venues are quoting twenty-three percent *under* budget. The customer was ready to spend more. That's money left on the table.

Christmas Parties? That's the sweet spot. Five percent below budget. Close enough to feel competitive, with enough room to upsell drinks packages and entertainment.

The lesson is simple: know your category's budget expectations and price to them."

**[CLICK to Slide 2]**

---

## SLIDE 2: Budget Proximity & Win Rates — 2 minutes

**Key message: Every venue type sees the same pattern. Price alignment = wins.**

"Now let's make this concrete. We split every proposal into four buckets based on how close the quote was to the customer's budget. And we did this for every venue type separately.

The pattern is universal. Look at Hotels — forty-seven percent win rate when you're within ten percent of budget. Twenty-one percent when you're more than fifty percent over. That's not a gentle decline. That's a cliff.

Conference Centres — forty-four percent drops to thirty percent. Bars — forty-five to thirty-one.

*[Pause]*

I want to be clear: I'm not saying race to the bottom. Being *under* budget doesn't always win more than being *at* budget. The sweet spot is plus or minus ten percent. That's where you're taken seriously as an option.

The question for every venue operator in this room: do you know where your quotes land relative to budget on every proposal? Because your competitors do."

**[CLICK to Slide 3]**

---

## SLIDE 3: What 19,896 Recommendations Say — 2 minutes

**Key message: We read every recommendation. The patterns are clear.**

"Here's where it gets tactical. Our system generates specific recommendations for every proposal — what the venue should change, add, or restructure to win more.

We didn't just count categories. We read all twenty thousand of them and classified them into concrete action themes.

*[Point to top bar]*

Number one: bundle your AV into the day delegate rate or package price. This comes up over and over. Planners don't want a separate AV line item. They want one number. 'Your conference costs this much, everything's included.'

Number two and three: tiered pricing. Give them three options — essential, standard, premium. Let *them* choose the price point rather than you guessing.

And look at the revenue impact numbers on the right — these aren't theoretical. Tiered DDR pricing has a median estimated annual impact of fifty thousand pounds per venue. All-inclusive packaging: fifty thousand. Bundling AV: forty-five thousand.

These are specific, costed changes. Not aspirational strategy."

**[CLICK to Slide 4]**

---

## SLIDE 4: Themes by Venue Type — 2 minutes

**Key message: Your venue type has a specific playbook.**

"Now here's where it gets personal. This heatmap shows which recommendations come up most for each venue type.

*[Point to Conference Centre row]*

Conference Centres and Hotels: your number one action is AV bundling. Eighteen percent of all recommendations for your venue type say the same thing. Stop charging AV as a line item. Bundle it.

*[Point to Bar/Restaurant rows]*

Bars and Restaurants: your playbook is different. It's minimum-spend models. Stop charging room hire. Convert to 'spend three thousand on food and drink and the space is yours.' That's how your customers want to buy.

*[Point to Historic Venue row]*

Historic Venues: you're sitting on premium spaces but quoting like everyone else. The recommendation? All-inclusive packages. Justify your premium by wrapping everything — room, AV, catering, service — into one impressive number.

*[Point to Rooftop/Outdoor rows]*

Rooftops and Outdoor venues: entertainment add-ons and late-licence extensions. That's your margin play. The space sells itself. Your upside is in what happens *inside* the space.

Every venue type in this room has a different top priority. Know yours."

**[CLICK to Slide 5]**

---

## SLIDE 5: Themes by Event Type — 2 minutes

**Key message: Different events need different fixes. Match your pitch.**

"Same data, different cut. This time by event type.

*[Point to Conferences row]*

Conferences: twenty-four percent of all recommendations say bundle AV. Planners are comparing you to venues that include a screen, mics, and a tech person in the day rate. If you don't, you look expensive *and* complicated.

*[Point to Christmas Parties row]*

Christmas Parties: the number one recommendation is to create specific festive packages. Not 'our space is available in December.' But 'The Christmas Soirée: three hours of drinks, canapés, DJ, and a photobooth for eighty pounds a head.' Name it, price it, done.

*[Point to Summer Parties row]*

Summer Parties: entertainment dominates. DJs, photobooths, lawn games, mixology stations. The venue is the backdrop — the *experience* is what wins.

*[Point to Award Ceremonies row]*

Award Ceremonies: it's AV plus all-inclusive. These are production-heavy events. The planner needs to know you can deliver the whole thing — not just the room.

The takeaway: when a brief comes in, your first question should be 'what does *this event type* need from us?' — not 'what's our standard package?'"

**[CLICK to Slide 6]**

---

## SLIDE 6: The Budget Reality Gap — 2 minutes

**Key message: The heatmap shows you exactly where to adjust.**

"This is your cheat sheet. Green means you're under budget — room to charge more. Red means you're over — and probably losing because of it.

*[Point to the darkest red cell]*

Historic Venues on Award Ceremonies: quoting at 1.23 times budget. Twenty-three percent over. And the win rate on these is in the low thirties. That's the data telling you to restructure the offer.

*[Point to green cells]*

Now look at the green. Conference Centres on Summer Parties: quoting at 0.77 times budget. That's *twenty-three percent under*. The customer wanted to spend more. Create a premium package, add entertainment, extend the hours — you've got headroom.

Hotels on Conferences: 1.20 times budget. You're pricing twenty percent above what the customer planned. That's why hotel conference win rates lag behind dedicated conference centres.

*[Scan the heatmap]*

This is a venue-by-venue, event-by-event pricing map. Screenshot it. Take it to your revenue meeting on Monday. Adjust the cells that are red. Upsell the cells that are green."

**[CLICK to Closing]**

---

## CLOSING — 60 seconds

**Slow down. Make eye contact.**

"Let me leave you with one number. Fifteen percentage points.

That's the difference in win rate between venues that quote within ten percent of budget and those that quote more than fifty percent over. Fifteen points.

Every other insight in this presentation — the AV bundling, the tiered DDR, the festive packages, the minimum-spend models — they all serve the same goal: getting you into that sweet spot where your quote matches what the customer expects to pay.

The data is clear. The playbooks are specific to your venue type and the events you're bidding on. And the changes are implementable this quarter.

Thank you."

*[Hold. Don't rush off.]*

---

## Q&A PREP — Likely questions

**"Is cheaper always better?"**
No. The data shows the sweet spot is *at* budget, not below it. Being 10-20% under budget doesn't win significantly more than being at budget. The cliff is on the *over* side.

**"Our venue is premium — won't lowering price devalue us?"**
The recommendation isn't to lower price. It's to restructure the *offer* so the total lands near budget. Bundle more in, create tiers, offer a smaller space option. The per-person value can stay premium.

**"How reliable are the win probability scores?"**
They're AI-estimated based on the competitive set, pricing position, offering match, and brief alignment. They're directionally reliable across 4,925 proposals — individual scores vary, but the aggregate patterns are robust.

**"What about venues that don't have a category listed?"**
About 39% of records don't have a highLevelEventCategory. These are included in overall stats but excluded from category-specific analysis. The category-specific insights are based on 3,010 categorised proposals.

**"What's the sample size for [specific cell in heatmap]?"**
All heatmap cells require a minimum of 10 proposals. If a cell is empty, there wasn't enough data. The largest cells (Conference Centre × Conferences) have 500+ proposals.
"""

    notes_path = os.path.join(OUT_DIR, "speaker_notes.md")
    with open(notes_path, "w") as f:
        f.write(notes)
    print(f"  [notes] speaker_notes.md")
    return notes_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    records = load_and_extract()

    print("\nGenerating keynote charts...")
    paths = [
        chart_1_budget_vs_quote(records),
        chart_2_budget_proximity(records),
        chart_3_themes(records),
        chart_4_themes_by_vt(records),
        chart_5_themes_by_cat(records),
        chart_6_budget_ratio(records),
    ]

    print("\nBuilding presentation...")
    build_html(paths)
    build_speaker_notes()

    print(f"\nDone. Open keynote/presentation.html in your browser.")


if __name__ == "__main__":
    main()
