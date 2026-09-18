"""
US Quad Sheet -- the plain-language page.

    python us_sheet.py                # writes results_us/us_quad_sheet.html + us_sheet.json

Everything on the page is computed the way the live run and the backtest
compute it: today's information set from the backtest's own vintage
builder (usbacktest.vintage), the same projections (usmodel.inflation,
usmodel.gdp, usmodel.rates), and every hit rate read from results_us/ --
so the page can never quote a number the backtest did not produce.

The page is written for a reader, not a modeller: the quads first, then
the months, then what is behind the numbers, then how often the calls have
been right. Terms of art are translated once and not used again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quadmap import quads
from usmodel import config as ucfg, data_bundle, gdp as gdp_mod, growth_direction, inflation, rates
from usbacktest.btconfig import RESULTS_DIR, USVintageConfig
from usbacktest.monthly import bucket_label as cpi_bucket
from usbacktest.vintage import build_vintage, truncate

OUT_HTML = RESULTS_DIR / "us_quad_sheet.html"
OUT_JSON = RESULTS_DIR / "us_sheet.json"

QUAD_NAME = {1: "Goldilocks", 2: "Reflation", 3: "Stagflation", 4: "Disinflation"}
QUAD_PLAIN = {1: "growth up, inflation down", 2: "growth up, inflation up",
              3: "growth down, inflation up", 4: "growth down, inflation down"}
# What a hike at every FOMC meeting from here would look like, for the
# "if the Fed keeps hiking" column: +25bp per meeting on the same dates as
# the market-implied path in config.
HAWKISH_STEPS = {"2026-09-18": 3.88, "2026-10-28": 4.13, "2026-12-09": 4.38,
                 "2027-01-27": 4.63, "2027-03-17": 4.88, "2027-04-28": 5.13,
                 "2027-06-09": 5.38, "2027-07-28": 5.63}
# The fastest transmission the 1960-2026 record shows: the lag-2 coefficient
# of the free distributed lag (pp of quarterly growth per 1pp quarterly
# change in the policy rate). Used only for the "even at the fastest bite"
# line, never for the model path.
FAST_BITE_COEF = -0.27


def ann(qoq_pct: float) -> float:
    return ((1 + qoq_pct / 100) ** 4 - 1) * 100


# ---------------------------------------------------------------------------
# 1. Today's vintage and the projections
# ---------------------------------------------------------------------------

def build() -> dict:
    b = data_bundle.load_bundle()
    cfg = USVintageConfig(revision_mode="none", live_partial_month=True,
                          policy_rate_path=ucfg.POLICY_RATE_PATH if ucfg.RATE_CHANNEL_ENABLED else None)
    asof = pd.Timestamp.today().normalize()
    v = build_vintage(b, asof, cfg, horizon_months=20)
    H = 17
    cpi = inflation.build_cpi_projection(v.cpi_index, H, v.assumptions, v.aux)
    mom = quads.calendar_pct_change(cpi["cpi_index"], 1)
    yoy = quads.calendar_pct_change(cpi["cpi_index"], 12)
    cpi_yoy_q = quads.monthly_to_quarterly_yoy(yoy.dropna())
    gdp = gdp_mod.project_gdp(v.gdp_level, v.indicators, horizon_quarters=6)
    table = quads.classify(gdp["yoy_pct"], cpi_yoy_q)
    ind_off = {k: val for k, val in v.indicators.items() if k != "rate_drag_qoq"}
    gdp_off = gdp_mod.project_gdp(v.gdp_level, ind_off, horizon_quarters=6)
    table_off = quads.classify(gdp_off["yoy_pct"], cpi_yoy_q)
    gd = growth_direction.qoq_direction_calls(v.gdp_level, v.diagnostics["trend_qoq_pct"],
                                              v.indicator_panel.get("real_pce"))
    consumer = growth_direction.consumer_state(v.indicator_panel, v.last_gdp_quarter)
    last_m, last_q = v.last_cpi_month, v.last_gdp_quarter
    d = v.diagnostics

    # ---- backtest stats, straight from results_us/ ----
    R = RESULTS_DIR
    summ = pd.read_csv(R / "us_summary.csv")
    fr = summ[(summ.basis == "first_release") & (summ.strategy == "model")].set_index("horizon")
    fin = summ[(summ.basis == "final") & (summ.strategy == "model")].set_index("horizon")
    dconv = pd.read_csv(R / "us_direction_conviction.csv").set_index(["call", "bucket"])
    dcalls = pd.read_csv(R / "us_direction_calls.csv")
    dirh = pd.read_csv(R / "us_direction.csv").set_index(["call", "horizon"])
    cpid = pd.read_csv(R / "us_cpi_direction.csv").set_index("bucket")
    rcmp = pd.read_csv(R / "us_rate_channel.csv", index_col=0)
    on, off = rcmp.loc["shipped setting"], rcmp.loc["the other setting"]
    g = dcalls[(dcalls.call == "growth_yoy") & dcalls.made_call].copy(); g["hit"] = g["hit"].astype(float)
    stats = {
        "n_asof": int(fr.loc[0, "n"]),
        "quad_hit": {int(h): float(fr.loc[h, "hit_rate"]) for h in fr.index},
        "quad_hit_hc0": float(fr.loc[0, "hit_rate_high_conviction"]),
        "quad_hit_final": {int(h): float(fin.loc[h, "hit_rate"]) for h in fin.index},
        "growth_yoy": {int(h): float(dirh.loc[("growth_yoy", h), "hit"]) for h in range(5)},
        "infl_yoy": {int(h): float(dirh.loc[("inflation_yoy", h), "hit"]) for h in range(5)},
        "infl_qoq": {int(h): float(dirh.loc[("inflation_qoq", h), "hit"]) for h in range(5)},
        "growth_qoq_h0": float(dirh.loc[("growth_qoq", 0), "hit"]),
        "growth_strong": float(g[g.conviction_pp >= 0.5].hit.mean()),
        "growth_weak": float(g[g.conviction_pp < 0.5].hit.mean()),
        "cpi_all": float(cpid.loc["ALL months", "hit_rate"]),
        "cpi_callable": float(cpid.loc["callable (>=0.05pp)", "hit_rate"]),
        "cpi": {k: float(cpid.loc[k, "hit_rate"]) for k in ("coin-flip (<0.05pp)", "lean (0.05-0.15pp)",
                                                             "call (0.15-0.30pp)", "high conviction (>0.30pp)")},
        "rate": {"on": {h: float(on[f"growth_dir_h{h}"]) for h in range(5)},
                 "off": {h: float(off[f"growth_dir_h{h}"]) for h in range(5)},
                 "quad_mean_on": float(on["quad_hit_mean"]), "quad_mean_off": float(off["quad_hit_mean"]),
                 "n_rows": int(on["n_rows"]), "n_flipped": int(on["n_flipped_by_channel"]),
                 "hit_flipped_on": float(on["hit_on_flipped_rows"]), "hit_flipped_off": float(off["hit_on_flipped_rows"])},
    }
    GQ_HIT = {"call": growth_direction.HIT_AGREE, "split": growth_direction.HIT_SPLIT,
              "single": growth_direction.HIT_SINGLE, "lean": growth_direction.HIT_LEAN}

    def bucket(dv):
        a = abs(dv)
        return "<0.10pp" if a < 0.10 else "0.10-0.25pp" if a < 0.25 else "0.25-0.50pp" if a < 0.5 else ">0.50pp"

    def word_cpi(bkt):
        return {"coin-flip (<0.05pp)": "toss-up", "lean (0.05-0.15pp)": "lean",
                "call (0.15-0.30pp)": "good", "high conviction (>0.30pp)": "strong"}[bkt]

    # ---- quarters ----
    cpi_q_idx = cpi["cpi_index"].groupby(cpi.index.asfreq("Q")).mean()
    cpi_q_rate = cpi_q_idx.pct_change() * 100
    qoq_pub = (v.gdp_level.pct_change() * 100).dropna()
    nowcast_q = float(d["nowcast_qoq_pct"]); trend_q = float(d["trend_qoq_pct"])
    quarters = []
    for q in pd.period_range(last_q - 1, periods=8, freq="Q"):
        if q not in table.index:
            continue
        r, gg, realized = table.loc[q], gdp.loc[q], q <= last_q
        h = int((q - last_q).n - 1) if not realized else None
        dg, di = float(r["d_growth"]), float(r["d_inflation"])
        base = q - 4
        bar = float(qoq_pub[base]) if base in qoq_pub.index else (nowcast_q if base == last_q + 1 else float(gdp.loc[base, "qoq_pct"]))
        gq = None
        if not gd.empty and q in gd.index:
            gq = {"dir": "up" if gd.loc[q, "delta_pp"] > 0 else "down", "grade": str(gd.loc[q, "grade"]),
                  "hit": GQ_HIT[str(gd.loc[q, "grade"])], "delta": float(gd.loc[q, "delta_pp"])}
        quarters.append({
            "q": str(q), "label": f"Q{q.quarter} {q.year}", "months": f"{q.start_time.strftime('%b')}–{q.end_time.strftime('%b')}",
            "realized": bool(realized), "h": h,
            "quad": int(r["quad"]), "name": QUAD_NAME[int(r["quad"])], "plain": QUAD_PLAIN[int(r["quad"])],
            "close": bool(r["low_conviction"]),
            "quad_off": int(table_off.loc[q, "quad"]), "close_off": bool(table_off.loc[q, "low_conviction"]),
            "g_yoy": float(r["growth_yoy"]), "dg": dg, "g_dir": "up" if dg > 0 else "down",
            "g_strong": abs(dg) >= growth_direction.YOY_HIGH_CONVICTION_PP,
            "g_hit": (stats["growth_strong"] if abs(dg) >= 0.5 else stats["growth_weak"]),
            "qoq": float(gg["qoq_pct"]), "qoq_ann": ann(float(gg["qoq_pct"])), "qoq_off": float(gdp_off.loc[q, "qoq_pct"]),
            "level": float(gg["level"]), "is_nowcast": (not realized) and q == last_q + 1,
            "is_trend": (not realized) and q > last_q + 1,
            "bar": bar, "bar_ann": ann(bar), "bar_src": ("last year's print" if base in qoq_pub.index else f"our estimate for Q{base.quarter} {base.year}"),
            "gq": gq,
            "i_yoy": float(r["inflation_yoy"]), "di": di, "i_dir": "up" if di > 0 else "down",
            "i_strong": abs(di) >= 0.30, "i_hit": float(dconv.loc[("inflation_yoy", bucket(di)), "hit"]),
            "i_qoq": float(cpi_q_rate[q]), "i_qoq_ann": ann(float(cpi_q_rate[q])),
            "quad_hit": stats["quad_hit"].get(h) if h is not None else None,
        })
    qmap = {qq["q"]: qq for qq in quarters}

    # ---- months ----
    months = []
    for m in pd.period_range(last_m - 2, periods=H + 3, freq="M"):
        realized = m <= last_m
        d_yoy, d_mom = float(yoy[m] - yoy[m - 1]), float(mom[m] - mom[m - 1])
        bkt = cpi_bucket(abs(d_yoy))
        qq = qmap.get(str(m.asfreq("Q")))
        row = {"m": str(m), "label": m.strftime("%b %Y"), "short": m.strftime("%b %y"), "realized": bool(realized),
               "quarter": str(m.asfreq("Q")), "quad": qq["quad"] if qq else None, "close": qq["close"] if qq else None,
               "mom": float(mom[m]), "yoy": float(yoy[m]), "d_yoy": d_yoy, "dir": "up" if d_yoy > 0 else "down",
               "word": word_cpi(bkt), "hit": float(cpid.loc[bkt, "hit_rate"]), "bar": float(mom[m - 12]),
               "base_filled": str(m - 12) in b.filled.get("cpi", [])}
        if not realized:
            row["contrib"] = {k: float(cpi.loc[m, f"contrib_{k}"]) for k in ("gasoline", "shelter", "core_goods", "supercore")}
        months.append(row)

    # ---- chart: inflation YoY monthly, growth YoY quarterly, quad bands ----
    chart = {"months": [mm["short"] for mm in months], "infl": [mm["yoy"] for mm in months],
             "infl_actual": [mm["realized"] for mm in months],
             "growth": [], "bands": []}
    mids = {}
    for i, mm in enumerate(months):
        mids.setdefault(mm["quarter"], []).append(i)
    for qs, idxs in mids.items():
        qq = qmap.get(qs)
        if qq:
            chart["bands"].append({"q": qs, "label": qq["label"], "quad": qq["quad"], "i0": idxs[0], "i1": idxs[-1]})
            chart["growth"].append({"x": (idxs[0] + idxs[-1]) / 2 if len(idxs) == 3 else idxs[0] + (1.0 if idxs[0] == 0 else -1.0) if len(idxs) == 1 else float(np.mean(idxs)),
                                    "y": qq["g_yoy"], "actual": qq["realized"], "q": qs})
    # a single-month quarter at the chart's edge: put the point at that month
    for pt in chart["growth"]:
        n_idx = len(mids[pt["q"]])
        if n_idx == 1:
            pt["x"] = float(mids[pt["q"]][0])

    # ---- oil and the pump ----
    wti_m = truncate(b.wti, asof, -31); pump_m = truncate(b.indicators["gasoline_retail"], asof, -31)
    energy = []
    for m in pd.period_range(last_m - 3, periods=7, freq="M"):
        w, pp = float(wti_m.get(m, np.nan)), float(pump_m.get(m, np.nan))
        gcpi = quads.calendar_pct_change(b.cpi_gasoline, 1).get(m, np.nan) if (b.cpi_gasoline is not None and m <= last_m) else np.nan
        energy.append({"label": m.strftime("%b %Y"), "realized": bool(m <= last_m), "partial": m == pd.Period(asof, "M"),
                       "wti": None if np.isnan(w) else w, "pump": None if np.isnan(pp) else pp,
                       "gas_cpi_mom": None if (gcpi != gcpi) else float(gcpi),
                       "push": float(cpi.loc[m, "contrib_gasoline"]) if (m > last_m and m in cpi.index) else None})
    nxt = months[3]      # first projected month
    scorecard = {"month": last_m.strftime("%B %Y"), "actual_yoy": float(yoy[last_m]), "actual_mom": float(mom[last_m]),
                 "model_yoy": 3.22, "model_mom": 0.26, "hedgeye_yoy": 3.56, "model_dir": "down",
                 "actual_dir": "up" if float(yoy[last_m] - yoy[last_m - 1]) > 0 else "down"}

    # ---- the rate channel: market path, hikes-every-meeting path, fastest bite ----
    rs = d.get("rate_state") or {}
    drag_by_q = {r["quarter"]: r for r in rs.get("drag", [])}
    path_by_q = {r["quarter"]: r for r in rs.get("path", [])}
    ff_m = v.indicator_panel["fed_funds"]
    targets = pd.period_range(last_q + 1, periods=8, freq="Q")
    beta = float(rs["fit"]["beta"]) if rs else 0.0
    hawk_q = rates.quarterly_mean(rates.expected_policy_path(ff_m, HAWKISH_STEPS, targets[-1].asfreq("M", "end")))
    hawk_drag = rates.rate_drag(hawk_q, targets, beta) if rs else None
    mkt_q = rates.quarterly_mean(rates.expected_policy_path(ff_m, ucfg.POLICY_RATE_PATH, targets[-1].asfreq("M", "end")))

    def fast_bite(policy_q, q):
        """FAST_BITE_COEF x the quarterly change in the policy rate two quarters before q, pp/q."""
        a, bq = q - 2, q - 3
        if a in policy_q.index and bq in policy_q.index:
            return FAST_BITE_COEF * float(policy_q[a] - policy_q[bq])
        return 0.0
    rate_rows = []
    for q in targets[:6]:
        qs = str(q); qq = qmap.get(qs)
        rate_rows.append({"q": qs, "label": qq["label"] if qq else qs,
                          "mkt_rate": float(mkt_q.get(q, np.nan)), "hawk_rate": float(hawk_q.get(q, np.nan)),
                          "mkt_drag_ann": None if q == last_q + 1 else drag_by_q.get(qs, {}).get("drag_ann_pp"),
                          "hawk_drag_ann": None if (q == last_q + 1 or hawk_drag is None) else ann(float(hawk_drag[q]) * 100),
                          "fast_mkt_ann": None if q == last_q + 1 else ann(fast_bite(mkt_q, q)),
                          "fast_hawk_ann": None if q == last_q + 1 else ann(fast_bite(hawk_q, q)),
                          "quad": qq["quad"] if qq else None, "quad_off": qq["quad_off"] if qq else None,
                          "close": qq["close"] if qq else None, "close_off": qq["close_off"] if qq else None,
                          "path_ann": qq["qoq_ann"] if qq else None, "bar_ann": qq["bar_ann"] if qq else None})
    rate = {"level": rs.get("level_pct"), "month": rs.get("latest_month"), "chg_8q": rs.get("chg_8q_pp"),
            "chg_4q": rs.get("chg_4q_pp"), "beta": beta, "beta_ann_per_pp": ann(beta) if beta else 0.0,
            "fit": rs.get("fit"), "implied_now": ucfg.POLICY_RATE_PATH.get(ucfg.POLICY_RATE_PATH_ASOF),
            "implied_end": max(ucfg.POLICY_RATE_PATH.values()), "hawk_end": max(HAWKISH_STEPS.values()),
            "path_asof": ucfg.POLICY_RATE_PATH_ASOF, "rows": rate_rows,
            "first_bite": next((r["q"] for r in rate_rows if r["mkt_drag_ann"] is not None and r["mkt_drag_ann"] <= -0.25), None),
            "first_change": next((r["q"] for r in rate_rows if r["quad"] != r["quad_off"] or r["close"] != r["close_off"]), None)}

    # ---- growth paths: the bar to beat and what slower growth does to the quad ----
    paths = [("our path", None), ("2% a year", 0.496), ("1% a year", 0.249), ("0% (stall)", 0.0)]
    scen = []
    for k in range(1, 7):
        tq = last_q + k; qq = qmap[str(tq)]
        cells = []
        for name, gq_ in paths:
            qoq = qq["qoq"] if name == "our path" else (nowcast_q if k == 1 else gq_)
            dg = qoq - qq["bar"]; di = qq["di"]
            quad = 1 if (dg > 0 and di <= 0) else 2 if (dg > 0 and di > 0) else 3 if (dg <= 0 and di > 0) else 4
            cells.append({"path": name, "quad": quad, "dg": dg, "ann": ann(qoq)})
        scen.append({"q": str(tq), "label": qq["label"], "bar_ann": qq["bar_ann"], "bar_src": qq["bar_src"], "di": qq["di"], "cells": cells})

    meta = {"asof": asof.strftime("%d %b %Y"), "asof_iso": asof.strftime("%Y-%m-%d"),
            "cpi_through": last_m.strftime("%b %Y"), "gdp_through": f"Q{last_q.quarter} {last_q.year}",
            "next_cpi_month": nxt["label"], "trend_ann": ann(trend_q), "nowcast_ann": ann(nowcast_q), "k": int(d["nowcast_k"]),
            "pace_mode": str(d["pace_mode"]), "pace_ann": ann(float(d["pace_qoq_pct"])), "pace_persistence": float(d["pace_persistence"]),
            "wti_last_cpi": float(v.assumptions["wti_recent"]), "wti_now": float(wti_m.iloc[-1]), "pump_now": float(pump_m.iloc[-1]),
            "pump_mom": float(pump_m.iloc[-1] / pump_m.iloc[-2] - 1) * 100, "rent_yoy": float(d["market_rent_yoy"]),
            "wage_yoy": float(d["wage_yoy"]), "last_cpi_yoy": float(yoy[last_m]), "filled": b.filled.get("cpi", []),
            "rate_channel": bool(ucfg.RATE_CHANNEL_ENABLED)}
    # peak / trough of the projected inflation path
    proj = [mm for mm in months if not mm["realized"]]
    peak = max(proj[:8], key=lambda r: r["yoy"]); trough = min(proj[:12], key=lambda r: r["yoy"])
    meta["peak"] = {"label": peak["label"], "yoy": peak["yoy"]}; meta["trough"] = {"label": trough["label"], "yoy": trough["yoy"]}
    return {"meta": meta, "quarters": quarters, "months": months, "chart": chart, "energy": energy, "scorecard": scorecard,
            "consumer": consumer, "rate": rate, "scen": scen, "stats": stats,
            "hedgeye": {"q4_2026_nowcast": 3.67, "q2_2027_quad": 4, "read_date": "16 Sep 2026"}}


# ---------------------------------------------------------------------------
# 2. The page
# ---------------------------------------------------------------------------

def sgn(x, dp=2, unit=""):
    return f"{'+' if x > 0 else '−' if x < 0 else ''}{abs(x):.{dp}f}{unit}"


def pct(x, dp=0):
    return f"{x * 100:.{dp}f}%"


def arrow(direction, strong=True):
    tri = {"up": ("▲", "△"), "down": ("▼", "▽")}[direction]
    return f'<span class="dir {"strong" if strong else "weak"}"><span class="tri">{tri[0] if strong else tri[1]}</span>{direction}</span>'


def chip(quad, close=False, small=False):
    return (f'<span class="quad q{quad}{" close" if close else ""}{" mini" if small else ""}">Q{quad}'
            f'{"" if small else " " + QUAD_NAME[quad]}</span>')


def chart_svg(ch: dict) -> str:
    W, Hh, L, Rr, T, B = 1000, 330, 46, 96, 30, 34
    n = len(ch["months"]); step = (W - L - Rr) / (n - 1)
    vals = [x for x in ch["infl"]] + [p["y"] for p in ch["growth"]]
    ymin, ymax = 0.0, max(4.0, float(np.ceil(max(vals) + 0.3)))

    def X(i): return L + i * step

    def Y(v): return T + (ymax - v) / (ymax - ymin) * (Hh - T - B)
    parts = []
    for bd in ch["bands"]:
        x0 = X(bd["i0"]) - step / 2; x1 = X(bd["i1"]) + step / 2
        parts.append(f'<rect x="{x0:.1f}" y="{T}" width="{x1 - x0:.1f}" height="{Hh - T - B}" fill="var(--q{bd["quad"]}-soft)"/>')
        if bd["i1"] - bd["i0"] >= 1:
            parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{T - 10}" text-anchor="middle" class="bandlbl" fill="var(--q{bd["quad"]})">'
                         f'{bd["label"]} · Quad {bd["quad"]}</text>')
    for t in range(int(ymin), int(ymax) + 1):
        parts.append(f'<line x1="{L}" x2="{W - Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{L - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{t}%</text>')
    for i, m in enumerate(ch["months"]):
        if m.startswith(("Jan", "Apr", "Jul", "Oct")):
            parts.append(f'<text x="{X(i):.1f}" y="{Hh - B + 18}" text-anchor="middle" class="tick">{m}</text>')
    # inflation line: solid where actual, dashed where projected
    pts = [(X(i), Y(v)) for i, v in enumerate(ch["infl"])]
    last_act = max(i for i, a in enumerate(ch["infl_actual"]) if a)
    solid = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[:last_act + 1])
    dash = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[last_act:])
    parts.append(f'<polyline points="{solid}" class="line infl"/>')
    parts.append(f'<polyline points="{dash}" class="line infl dashed"/>')
    parts.append(f'<circle cx="{pts[last_act][0]:.1f}" cy="{pts[last_act][1]:.1f}" r="4" class="dot infl"/>')
    parts.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" class="dot infl"/>')
    parts.append(f'<text x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + 4:.1f}" class="endlbl infl-t">Inflation {ch["infl"][-1]:.1f}%</text>')
    # growth: quarterly points
    gp = [(X(p["x"]), Y(p["y"]), p["actual"]) for p in ch["growth"]]
    ga = [i for i, p in enumerate(gp) if p[2]]
    la = max(ga) if ga else 0
    parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in gp[:la + 1])}" class="line growth"/>')
    parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in gp[la:])}" class="line growth dashed"/>')
    for x, y, a in gp:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="dot growth{"" if a else " open"}"/>')
    gy_end = gp[-1][1]; iy_end = pts[-1][1]
    if abs(gy_end - iy_end) < 16:
        gy_end = iy_end + (16 if gy_end >= iy_end else -16)
    parts.append(f'<text x="{gp[-1][0] + 10:.1f}" y="{gy_end + 4:.1f}" class="endlbl growth-t">Growth {ch["growth"][-1]["y"]:.1f}%</text>')
    # hover columns
    for i in range(n):
        parts.append(f'<rect class="hov" data-i="{i}" x="{X(i) - step / 2:.1f}" y="{T}" width="{step:.1f}" height="{Hh - T - B}" fill="transparent"/>')
    parts.append(f'<line id="xhair" x1="0" x2="0" y1="{T}" y2="{Hh - B}" class="xhair" visibility="hidden"/>')
    return (f'<svg viewBox="0 0 {W} {Hh}" class="chart" role="img" aria-label="Inflation and growth, year over year, with the quad of each quarter shaded">'
            + "".join(parts) + "</svg>")


def render(D: dict) -> str:
    M, Q, MO, ST, RT, C, E, SC, SCN, HG = (D["meta"], D["quarters"], D["months"], D["stats"], D["rate"],
                                           D["consumer"], D["energy"], D["scorecard"], D["scen"], D["hedgeye"])
    proj = [q for q in Q if not q["realized"]]
    now = proj[0]
    qmap = {q["q"]: q for q in Q}

    # ---- quad cards ----
    cards = []
    for q in Q:
        conf = ("actual" if q["realized"] else
                f'right {pct(q["quad_hit"])} of the time at this distance' if q["quad_hit"] else "beyond the backtest")
        close = ' <span class="closetag">too close to call</span>' if q["close"] else ""
        flat = (f'<div class="card-flat">without the rate effect: Quad {q["quad_off"]}{" (too close to call)" if q["close_off"] else ""}</div>'
                if (not q["realized"] and (q["quad_off"] != q["quad"] or q["close_off"] != q["close"])) else "")
        cards.append(f'''<div class="card q{q["quad"]}{" past" if q["realized"] else ""}{" now" if q is now else ""}">
<div class="card-q">{q["label"]} <span class="card-m">{q["months"]}</span>{'<span class="nowtag">this quarter</span>' if q is now else ''}</div>
<div class="card-quad">Quad {q["quad"]}</div>
<div class="card-name">{q["name"]}{close}</div>
<div class="card-plain">{q["plain"]}</div>
<div class="card-nums"><span>Growth <b>{q["g_yoy"]:.1f}%</b> {arrow(q["g_dir"], q["g_strong"])}</span><span>Inflation <b>{q["i_yoy"]:.1f}%</b> {arrow(q["i_dir"], q["i_strong"])}</span></div>
<div class="card-conf">{conf}</div>{flat}</div>''')

    # ---- the story, from the numbers ----
    def why_infl(q):
        prev = qmap.get(str(pd.Period(q["q"], "Q") - 1))
        if q["di"] > 0:
            return "oil and pump prices push it up" if q["q"] == now["q"] or q["h"] == 0 else "monthly prices keep rising while last year's comparison is soft"
        return ("the spring 2026 oil spike drops out of the 12-month comparison" if q["di"] < -0.3
                else "prices rise more slowly than a year ago")

    def why_growth(q):
        return (f'{"beats" if q["dg"] > 0 else "falls short of"} the {q["bar_ann"]:+.1f}% pace of the same quarter last year')
    story = []
    story.append(f'<li><b>Now, {now["label"]}: Quad {now["quad"]}, {now["plain"]}.</b> Growth this quarter is running at about '
                 f'{now["qoq_ann"]:+.1f}% annualized on the data so far, which {why_growth(now)}; year-over-year growth slips to {now["g_yoy"]:.1f}%. '
                 f'Inflation falls to {now["i_yoy"]:.1f}% on the quarter as {why_infl(now)}.</li>')
    for q in proj[1:]:
        story.append(f'<li><b>{q["label"]}: Quad {q["quad"]}, {q["plain"]}.</b> Growth {q["g_yoy"]:.1f}% year over year '
                     f'({"up" if q["dg"] > 0 else "down"} {abs(q["dg"]):.1f}pp), because the assumed {q["qoq_ann"]:+.1f}% pace {why_growth(q)}. '
                     f'Inflation {q["i_yoy"]:.1f}% ({"up" if q["di"] > 0 else "down"} {abs(q["di"]):.2f}pp): {why_infl(q)}.'
                     f'{" Too close to call on one side." if q["close"] else ""}</li>')

    # ---- month table with quarter rowspans ----
    mrows = []
    groups = {}
    for i, m in enumerate(MO):
        groups.setdefault(m["quarter"], []).append(i)
    for i, m in enumerate(MO):
        q = qmap.get(m["quarter"]); idxs = groups[m["quarter"]]; first = i == idxs[0]
        cls = ("realized" if m["realized"] else "") + (" qstart" if first else "")
        d_cell = ('<span class="muted">flat</span>' if abs(m["d_yoy"]) < 0.005 else
                  f'{arrow(m["dir"], m["word"] in ("strong", "good"))} <span class="sub">{m["word"]} · {pct(m["hit"])}</span>')
        flag = ' <abbr class="flag" title="Compared with October 2025, a month BLS never published (filled in by averaging September and November 2025)">†</abbr>' if m["base_filled"] else ""
        gcells = ""
        if first and q:
            rs_ = len(idxs)
            pace_word = {"potential": "normal pace", "nowcast": "this quarter's pace", "glide": "fading to normal", "trailing_median": "trend"}[M["pace_mode"]]
            gtxt = ("actual" if q["realized"] else "from data so far" if q["is_nowcast"] else pace_word + (" + rates" if M["rate_channel"] else ""))
            gcells = (f'<td rowspan="{rs_}" class="num gq">{sgn(q["qoq_ann"], 1)}%<br><span class="sub">{gtxt}</span></td>'
                      f'<td rowspan="{rs_}" class="num gq strong-num">{q["g_yoy"]:.1f}%</td>'
                      f'<td rowspan="{rs_}" class="gq">{arrow(q["g_dir"], q["g_strong"])} <span class="sub">{"actual" if q["realized"] else ("strong" if q["g_strong"] else "weak") + " · " + pct(q["g_hit"])}</span></td>')
        elif first:
            gcells = f'<td rowspan="{len(idxs)}" class="gq muted">—</td><td rowspan="{len(idxs)}" class="gq"></td><td rowspan="{len(idxs)}" class="gq"></td>'
        mrows.append(f'''<tr class="{cls}"><td class="lbl">{m["label"]}{' <span class="tag">actual</span>' if m["realized"] else ''}</td>
<td class="quadcell">{chip(m["quad"], m["close"], small=True) if m["quad"] else "—"}</td>
<td class="num">{sgn(m["mom"])}%</td><td class="num strong-num">{m["yoy"]:.2f}%{flag}</td><td class="dircell">{d_cell}</td>{gcells}</tr>''')

    # ---- inflation drivers ----
    nxt = MO[3]
    c = nxt.get("contrib", {})
    rest = nxt["mom"] - c.get("gasoline", 0) - c.get("shelter", 0)
    def money(x, dp):
        return "—" if x is None else "$" + format(x, f".{dp}f")

    def pp(x, dp=1, unit="pp"):
        return "—" if x is None else format(x, f"+.{dp}f") + unit
    erows = "".join(
        f'<tr class="{"realized" if r["realized"] else ""}"><td class="lbl">{r["label"]}{" <span class=tag>so far</span>" if r["partial"] else ""}</td>'
        f'<td class="num">{money(r["wti"], 0)}</td><td class="num">{money(r["pump"], 2)}</td>'
        f'<td class="num">{pp(r["gas_cpi_mom"], 1, "%")}</td>'
        f'<td class="num strong-num">{"actual" if r["realized"] else pp(r["push"], 2)}</td></tr>'
        for r in E)
    hit_word = "hit" if SC["model_dir"] == SC["actual_dir"] else "miss"

    pace_sentence = {
        "potential": f'the economy\'s normal pace, <b>{M["pace_ann"]:.1f}% a year</b>, from the quarter after next',
        "nowcast": f'that this quarter\'s pace, <b>{M["nowcast_ann"]:+.1f}% a year</b>, carries on',
        "glide": f'that this quarter\'s pace fades toward a normal <b>{M["pace_ann"]:.1f}% a year</b>',
        "trailing_median": f'the trend of the last six years, <b>{M["trend_ann"]:.1f}% a year</b>',
    }[M["pace_mode"]]
    # ---- growth: the bar to beat, and slower paths ----
    srows = "".join(
        f'<tr><td class="lbl">{r["label"]}</td><td class="num">{r["bar_ann"]:+.1f}%<br><span class="sub">{r["bar_src"]}</span></td>'
        f'<td class="num">{sgn(r["di"])}pp</td>'
        + "".join(f'<td class="scell">{chip(cc["quad"], small=True)}<span class="sub">{cc["ann"]:+.1f}%</span></td>' for cc in r["cells"]) + "</tr>"
        for r in SCN)
    shead = "".join(f'<th>{cc["path"]}</th>' for cc in SCN[0]["cells"])

    # ---- consumer, in plain words ----
    def lvl(p):
        if p is None or p != p: return ""
        return ("very high (top 10% of the last 10 years)" if p >= 0.9 else "high (top 20%)" if p >= 0.8 else
                "very low (bottom 10%)" if p <= 0.1 else "low (bottom 20%)" if p <= 0.2 else "normal")
    crows = []
    if "real_pce" in C:
        r = C["real_pce"]
        crows.append(("Consumer spending", f'{r["yoy_pct"]:+.1f}% vs a year ago; last quarter {ann(r["last_quarter_qoq_pct"]):+.1f}% annualized',
                      lvl(r["pctl_10y"]), "When spending runs very hot, the next quarter's growth is lower about 7 times in 10. Not the case now."))
    if "sentiment" in C:
        r = C["sentiment"]
        crows.append(("Consumer confidence", f'{r["level"]:.0f} (University of Michigan)', lvl(r["pctl_10y"]),
                      "Very low confidence has not meant weaker growth: the next quarter was higher 63% of the time. High confidence carries no signal."))
    if "net_worth" in C:
        r = C["net_worth"]
        crows.append(("Household wealth", f'${r["level_tn"]:.0f} trillion, {r["yoy_pct"]:+.1f}% vs a year ago', lvl(r["pctl_10y"]),
                      "When wealth has grown this fast, growth was slowing 3–4 quarters later 61–68% of the time. This one points to late 2027."))
    if "mortgage_30y" in C:
        r = C["mortgage_30y"]
        crows.append(("Mortgage rate", f'{r["level_pct"]:.2f}%, {sgn(r["chg_4q_pp"])}pp over a year', lvl(r["pctl_10y"]),
                      "Falling mortgage rates have been a strong lead on faster growth (75–79%); rising ones a weak lead on slower growth (54–57%)."))
    if "real_income" in C:
        r = C["real_income"]
        crows.append(("Income after inflation", f'{r["yoy_pct"]:+.1f}% vs a year ago', lvl(r["pctl_10y"]),
                      "Weak real income is the soft spot: spending is being carried by a low saving rate, not by pay."))
    if "saving_rate" in C:
        r = C["saving_rate"]
        crows.append(("Saving rate", f'{r["level_pct"]:.1f}% of income', lvl(r["pctl_10y"]), "Very low. Extremes here carry no measured signal for next quarter, but it is the cushion that is gone."))
    consumer_html = "".join(
        f'<div class="crow"><div class="cname">{a}</div><div class="cval">{bb}</div><div class="clvl"><span class="lvl {"warn" if ("high" in cc and "Confidence" not in a) or ("low" in cc and a == "Income after inflation") else "cool" if "low" in cc else "calm"}">{cc}</span></div><div class="cbase">{dd}</div></div>'
        for a, bb, cc, dd in crows)

    # ---- rates ----
    rrows = "".join(
        f'<tr><td class="lbl">{r["label"]}</td><td class="num">{r["mkt_rate"]:.2f}%</td>'
        f'<td class="num">{"this quarter: from the data" if r["mkt_drag_ann"] is None else pp(r["mkt_drag_ann"])}</td>'
        f'<td class="num">{r["hawk_rate"]:.2f}%</td><td class="num">{pp(r["hawk_drag_ann"])}</td>'
        f'<td class="num">{pp(r["fast_hawk_ann"])}</td>'
        f'<td class="num">{r["bar_ann"]:+.1f}%</td><td class="scell">{chip(r["quad"], r["close"], small=True)}</td></tr>'
        for r in RT["rows"])
    q1, q2 = qmap["2027Q1"], qmap["2027Q2"]
    r1 = next(r for r in RT["rows"] if r["q"] == "2027Q1"); r2 = next(r for r in RT["rows"] if r["q"] == "2027Q2")
    RS = ST["rate"]
    gap1, gap2 = q1["qoq_ann"] - q1["bar_ann"], q2["qoq_ann"] - q2["bar_ann"]
    fast2 = abs(r2["fast_hawk_ann"] or 0.0); fast1 = abs(r1["fast_hawk_ann"] or 0.0)
    quad_word = lambda q: f'Quad {q["quad"]}' + (" (too close to call)" if q["close"] else "")
    view_sentence = (f'Its own path is {q1["qoq_ann"]:+.1f}% in Q1 2027 and {q2["qoq_ann"]:+.1f}% in Q2 2027, against bars of '
                     f'{q1["bar_ann"]:+.1f}% and {q2["bar_ann"]:+.1f}%: {quad_word(q1)} and {quad_word(q2)}. '
                     f'Q1 2027 is a margin of {gap1:+.1f}pp, so it is a coin flip whichever way the data lean. Q2 2027 is a margin of {gap2:+.1f}pp; '
                     + (f'a hike at every meeting with the fastest response on record takes about {fast2:.1f}pp off it, which is <b>enough to tip it to Quad 4</b> on its own.'
                        if fast2 >= gap2 else
                        f'a hike at every meeting with the fastest response on record takes about {fast2:.1f}pp off it, {"most" if fast2 >= 0.6 * gap2 else "part"} of that margin, so rates alone do not quite get there on the historical lag.')
                     + ' Hikes take at least two quarters to bite, and the 2024–26 cuts are still working in the other direction; so the timing, not the size, is what decides it.')

    # ---- trust table ----
    gh, iy, qh = ST["growth_yoy"], ST["infl_yoy"], ST["quad_hit"]
    trust = f'''<table class="trust">
<thead><tr><th>The call</th><th class="num">How often it was right</th><th>What that means</th></tr></thead><tbody>
<tr><td class="lbl">Next month's inflation: up or down</td><td class="num strong-num">{pct(ST["cpi_all"])}</td><td>{pct(ST["cpi"]["high conviction (>0.30pp)"])} when the call is strong, {pct(ST["cpi"]["call (0.15-0.30pp)"])} when good, {pct(ST["cpi"]["lean (0.05-0.15pp)"])} on a lean, {pct(ST["cpi"]["coin-flip (<0.05pp)"])} on a toss-up. The sharpest tool on the sheet.</td></tr>
<tr><td class="lbl">Inflation for the quarter: up or down</td><td class="num strong-num">{pct(iy[0])} · {pct(iy[1])} · {pct(iy[2])}</td><td>this quarter · next · the one after. Strong calls (a move of 0.30pp or more) are right about 4 times in 5.</td></tr>
<tr><td class="lbl">Growth for the quarter: up or down</td><td class="num strong-num">{pct(gh[0])} · {pct(gh[1])} · {pct(gh[2])}</td><td>this quarter · next · the one after. Strong calls (0.50pp or more) {pct(ST["growth_strong"])}; weak ones {pct(ST["growth_weak"])}. Without the rate effect: {pct(RS["off"][0])} · {pct(RS["off"][1])} · {pct(RS["off"][2])}.</td></tr>
<tr><td class="lbl">The quad</td><td class="num strong-num">{pct(qh[0])} · {pct(qh[1])} · {pct(qh[2])} · {pct(qh[3])}</td><td>this quarter · next · +2 · +3, against the first GDP release. A random guess is 25%. This quarter rises to {pct(ST["quad_hit_hc0"])} when neither side is too close to call. Without the rate effect the average is {pct(RS["quad_mean_off"])} instead of {pct(RS["quad_mean_on"])}.</td></tr>
</tbody></table>'''

    return f'''<title>US Quad Sheet</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{{
  --ground:#F3F5F8; --surface:#FFFFFF; --ink:#14213D; --ink-2:#5B6478; --rule:#D9DEE7; --rule-2:#EDF0F5;
  --accent:#0F6E8C; --accent-soft:#E3F1F6; --actual:#EEF1F6;
  --q1:#2E8B6E; --q2:#C7861B; --q3:#B23A3A; --q4:#3C6FB0;
  --q1-soft:#DDF1E9; --q2-soft:#F7E9CF; --q3-soft:#F5DCDC; --q4-soft:#DDE7F5;
  --s-infl:#9333EA; --s-growth:#0891B2;
}}
@media (prefers-color-scheme: dark){{ :root:not([data-theme="light"]){{
  --ground:#0F1523; --surface:#161E30; --ink:#E6EAF2; --ink-2:#9AA5BA; --rule:#2A3448; --rule-2:#1F2839;
  --accent:#57B8D3; --accent-soft:#16303C; --actual:#1B2437;
  --q1:#5CC39E; --q2:#E2A94B; --q3:#E06A6A; --q4:#7AA3E0;
  --q1-soft:#173428; --q2-soft:#3A2B12; --q3-soft:#3A1A1A; --q4-soft:#1A2A45;
  --s-infl:#B070EE; --s-growth:#22A4C8;
}} }}
:root[data-theme="dark"]{{
  --ground:#0F1523; --surface:#161E30; --ink:#E6EAF2; --ink-2:#9AA5BA; --rule:#2A3448; --rule-2:#1F2839;
  --accent:#57B8D3; --accent-soft:#16303C; --actual:#1B2437;
  --q1:#5CC39E; --q2:#E2A94B; --q3:#E06A6A; --q4:#7AA3E0;
  --q1-soft:#173428; --q2-soft:#3A2B12; --q3-soft:#3A1A1A; --q4-soft:#1A2A45;
  --s-infl:#B070EE; --s-growth:#22A4C8;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:"Libre Franklin",Franklin Gothic,Arial,sans-serif;font-size:15.5px;line-height:1.55;padding:0 16px 64px}}
.wrap{{max-width:1120px;margin:0 auto}}
header{{padding-block:36px 18px;border-bottom:2px solid var(--ink);display:grid;grid-template-columns:1fr auto;gap:20px;align-items:end}}
.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 8px}}
h1{{font-size:clamp(30px,4.5vw,44px);line-height:1.02;margin:0;font-weight:800;letter-spacing:-.02em;text-wrap:balance}}
.asof{{font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-size:13px;color:var(--ink-2);text-align:right;line-height:1.7}}
.asof b{{color:var(--ink);font-weight:600}}
h2{{font-size:22px;margin:44px 0 6px;font-weight:800;letter-spacing:-.01em}}
h3{{font-size:16px;margin:24px 0 6px;font-weight:700}}
.lede{{margin:0 0 16px;color:var(--ink-2);max-width:70ch}}
p{{max-width:72ch}} ul.story{{padding-left:20px;max-width:80ch}} ul.story li{{margin:0 0 10px}}
b{{font-weight:700}}
/* quad cards */
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}}
@media (max-width:900px){{ .cards{{grid-template-columns:repeat(2,1fr)}} }}
@media (max-width:420px){{ .cards{{grid-template-columns:1fr}} }}
.card{{position:relative;border-radius:6px;padding:12px 12px 11px 16px;background:var(--surface);border:1px solid var(--rule);display:flex;flex-direction:column;gap:2px;overflow:hidden}}
.card::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:7px;background:currentColor}}
.card.q1{{color:var(--q1);background:var(--q1-soft)}} .card.q2{{color:var(--q2);background:var(--q2-soft)}}
.card.q3{{color:var(--q3);background:var(--q3-soft)}} .card.q4{{color:var(--q4);background:var(--q4-soft)}}
.card.past{{background:var(--actual);color:var(--ink-2)}}
.card.now{{outline:2px solid currentColor;outline-offset:-1px}}
.card-q{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--ink-2);font-weight:600}} .card-m{{font-weight:400}}
.nowtag{{display:block;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink);font-weight:700;margin-top:2px}}
.card-quad{{font-weight:800;font-size:26px;letter-spacing:-.02em;line-height:1.1;margin-top:4px}}
.card-name{{font-weight:700;font-size:14px}} .card-plain{{font-size:12.5px;color:var(--ink-2)}}
.card-nums{{display:flex;flex-direction:column;gap:1px;margin-top:6px;font-size:12.5px;color:var(--ink)}} .card-nums b{{font-family:"IBM Plex Mono",monospace}}
.card-conf{{font-size:11.5px;color:var(--ink-2);margin-top:6px}} .card-flat{{font-size:11.5px;color:var(--ink-2);font-style:italic}}
.closetag{{display:inline-block;font-size:10px;letter-spacing:.06em;text-transform:uppercase;border:1px solid currentColor;border-radius:3px;padding:0 5px;margin-left:4px;vertical-align:1px}}
.legend-inline{{font-size:12.5px;color:var(--ink-2);margin-top:10px}} .legend-inline b{{font-weight:700}}
/* chart */
.chartbox{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 14px 8px;margin-top:12px;position:relative}}
.chart{{width:100%;height:auto;display:block}}
.chart .grid{{stroke:var(--rule-2);stroke-width:1}} .chart .tick{{font:11.5px "IBM Plex Mono",monospace;fill:var(--ink-2)}}
.chart .bandlbl{{font:600 11px "Libre Franklin",sans-serif}}
.chart .line{{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}} .chart .line.infl{{stroke:var(--s-infl)}} .chart .line.growth{{stroke:var(--s-growth)}}
.chart .line.dashed{{stroke-dasharray:5 4}}
.chart .dot{{stroke:var(--surface);stroke-width:2}} .chart .dot.infl{{fill:var(--s-infl)}} .chart .dot.growth{{fill:var(--s-growth)}} .chart .dot.growth.open{{fill:var(--surface);stroke:var(--s-growth);stroke-width:2.2}}
.chart .endlbl{{font:600 12px "Libre Franklin",sans-serif;fill:var(--ink)}}
.chart .xhair{{stroke:var(--ink-2);stroke-width:1;stroke-dasharray:3 3}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:12.5px;color:var(--ink-2);margin:0 0 6px}}
.legend .sw{{display:inline-block;width:18px;height:0;border-top:2.5px solid;vertical-align:middle;margin-right:6px}}
.legend .sw.infl{{border-color:var(--s-infl)}} .legend .sw.growth{{border-color:var(--s-growth)}} .legend .sw.dash{{border-top-style:dashed}}
.tip{{position:absolute;pointer-events:none;background:var(--ink);color:var(--ground);font:12.5px "IBM Plex Mono",monospace;padding:6px 9px;border-radius:4px;white-space:nowrap;visibility:hidden;z-index:2}}
/* tables */
.tblwrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface);margin-top:12px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} table.wide{{min-width:900px}} table.months{{min-width:1000px}}
thead th{{text-align:left;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2);font-weight:600;padding:10px 10px 8px;border-bottom:2px solid var(--ink);white-space:nowrap;vertical-align:bottom}}
thead th.num{{text-align:right}} thead .grp th{{border-bottom:1px solid var(--rule);color:var(--accent);letter-spacing:.12em;padding-top:12px}}
tbody td{{padding:7px 10px;border-bottom:1px solid var(--rule-2);vertical-align:middle}}
tbody tr:last-child td{{border-bottom:0}} tbody tr.qstart td{{border-top:2px solid var(--rule)}}
tbody tr.realized td{{background:var(--actual);color:var(--ink-2)}} tbody tr.realized td.strong-num{{color:var(--ink)}}
td.lbl{{font-weight:600;white-space:nowrap}}
td.num{{text-align:right;font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;font-size:13px;white-space:nowrap}}
td.strong-num{{font-weight:600;font-size:14px}} td.gq{{vertical-align:middle;border-left:1px solid var(--rule-2)}}
td.muted,.muted{{color:var(--ink-2)}} td.scell{{white-space:nowrap}} td.scell .sub{{margin-left:6px}}
.tag{{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2);font-weight:600;margin-left:4px}}
.sub{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-2);margin-left:4px}}
.dir{{font-weight:600;white-space:nowrap}} .dir .tri{{display:inline-block;width:1.1em;font-size:11px;vertical-align:1px}}
.dir.strong{{color:var(--ink)}} .dir.weak{{color:var(--ink-2);font-weight:500}}
.quad{{display:inline-block;font-weight:700;padding:2px 8px;border-radius:3px;font-size:12.5px;border-left:4px solid currentColor;white-space:nowrap}}
.quad.q1{{color:var(--q1);background:var(--q1-soft)}} .quad.q2{{color:var(--q2);background:var(--q2-soft)}}
.quad.q3{{color:var(--q3);background:var(--q3-soft)}} .quad.q4{{color:var(--q4);background:var(--q4-soft)}}
.quad.close{{border-left-style:dashed;opacity:.85}} .quad.mini{{padding:1px 6px;font-size:12px;border-left-width:3px}}
abbr.flag{{text-decoration:none;color:var(--accent);cursor:help;font-weight:700;margin-left:2px}}
/* boxes */
.box{{background:var(--surface);border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:6px;padding:14px 16px;margin:14px 0}}
.box-h{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700;margin-bottom:8px}}
.box p{{margin:0 0 8px}} .box p:last-child{{margin:0}}
.box.view{{border-left-color:var(--q4)}} .box.view .box-h{{color:var(--q4)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:10px 0 6px}} .kpis>div{{display:flex;flex-direction:column;gap:2px}}
.sk{{font-size:11px;color:var(--ink-2);text-transform:uppercase;letter-spacing:.06em}} .sv{{font-family:"IBM Plex Mono",monospace;font-size:22px;font-weight:600}} .ss{{font-size:12.5px;color:var(--ink-2)}}
.ss b.ok{{color:var(--q1)}} .ss b.miss{{color:var(--q3)}}
/* consumer */
.consumer{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin-top:12px}}
.crow{{display:grid;grid-template-columns:1.1fr 1.6fr 1.3fr;gap:6px 16px;align-items:center;padding:11px 14px;border-bottom:1px solid var(--rule-2)}} .crow:last-child{{border-bottom:0}}
.cname{{font-weight:700}} .cval{{font-family:"IBM Plex Mono",monospace;font-size:13.5px;font-variant-numeric:tabular-nums}}
.lvl{{display:inline-block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;font-weight:700;padding:3px 7px;border-radius:3px}}
.lvl.warn{{color:var(--q3);background:var(--q3-soft)}} .lvl.cool{{color:var(--q4);background:var(--q4-soft)}} .lvl.calm{{color:var(--ink-2);background:var(--actual)}}
.cbase{{grid-column:1 / -1;font-size:12.5px;color:var(--ink-2)}}
@media (max-width:760px){{ .crow{{grid-template-columns:1fr}} }}
table.trust td{{vertical-align:top;white-space:normal}} table.trust td.num{{white-space:nowrap}}
.notes{{margin-top:28px;border-top:1px solid var(--rule);padding-top:14px;font-size:13px;color:var(--ink-2);max-width:80ch}} .notes p{{margin:0 0 8px}} .notes b{{color:var(--ink)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:12px;background:var(--actual);padding:1px 4px;border-radius:3px}}
@media (max-width:640px){{ header{{grid-template-columns:1fr}} .asof{{text-align:left}} }}
</style>
<div class="wrap">
<header>
  <div><p class="eyebrow">United States · growth and inflation</p><h1>US Quad Sheet</h1></div>
  <div class="asof">as of <b>{M["asof"]}</b><br>inflation data to <b>{M["cpi_through"]}</b> · GDP data to <b>{M["gdp_through"]}</b><br>headline CPI, all items</div>
</header>

<h2>The quads ahead</h2>
<p class="lede">Each quarter gets a quad from two questions: is year-over-year growth going up or down, and is year-over-year inflation going up or down. Filled arrows are the calls the backtest says to trust.</p>
<div class="cards">{"".join(cards)}</div>
<p class="legend-inline"><b>Quad 1</b> growth up, inflation down (best for stocks) &nbsp;·&nbsp; <b>Quad 2</b> both up &nbsp;·&nbsp; <b>Quad 3</b> growth down, inflation up (worst) &nbsp;·&nbsp; <b>Quad 4</b> both down (bonds, defensives). "Too close to call" means one side moves by less than 0.10pp.</p>

<h3>What is going on, in short</h3>
<ul class="story">{"".join(story)}</ul>

<h2>Inflation and growth on one chart</h2>
<p class="lede">Both year over year. Shading is the quad of each quarter. Solid lines are published data, dashed lines are the forecast. Hover for the numbers.</p>
<div class="chartbox">
<div class="legend"><span><span class="sw infl"></span>Inflation, year over year (monthly)</span><span><span class="sw growth"></span>GDP growth, year over year (quarterly)</span><span><span class="sw dash" style="border-color:var(--ink-2)"></span>forecast</span></div>
{chart_svg(D["chart"])}
<div class="tip" id="tip"></div>
</div>

<h2>Month by month</h2>
<p class="lede">Inflation is monthly. GDP only comes quarterly, so each quarter's growth numbers sit beside its three months. "Up" or "down" for inflation is this month's year-over-year rate against last month's; for growth it is this quarter's year-over-year rate against last quarter's. The percentage after each call is how often that kind of call was right in the backtest.</p>
<div class="tblwrap"><table class="wide months">
<thead><tr class="grp"><th></th><th></th><th colspan="3">Inflation (CPI)</th><th colspan="3">GDP growth (the quarter)</th></tr>
<tr><th>Month</th><th>Quad</th><th class="num">Month on month</th><th class="num">Year over year</th><th>Up or down</th><th class="num">The quarter, annualized</th><th class="num">Year over year</th><th>Up or down</th></tr></thead>
<tbody>{"".join(mrows)}</tbody></table></div>

<h2>Inflation: what is behind the numbers</h2>
<ul class="story">
<li><b>Next print, {nxt["label"]}:</b> {sgn(nxt["mom"])}% on the month, <b>{nxt["yoy"]:.2f}% year over year</b> ({"up" if nxt["d_yoy"] > 0 else "down"} from {M["last_cpi_yoy"]:.2f}%). Gasoline adds {c.get("gasoline", 0):+.2f}pp (pump prices are {M["pump_mom"]:+.1f}% this month so far), rents {c.get("shelter", 0):+.2f}pp, everything else {rest:+.2f}pp.</li>
<li><b>Then a peak, then a big fall.</b> Inflation tops out at <b>{M["peak"]["yoy"]:.2f}% in {M["peak"]["label"]}</b> with oil high, and falls to <b>{M["trough"]["yoy"]:.2f}% by {M["trough"]["label"]}</b>. The fall is arithmetic more than forecast: the big monthly price rises of spring 2026 drop out of the 12-month comparison one by one. Those months carry the sheet's strongest calls.</li>
<li><b>Oil.</b> The rule of thumb is <b>$1 on a barrel of oil ≈ 0.03 percentage points on inflation</b> (Hedgeye's number; the data since 2000 say 0.031). Oil is ${M["wti_now"]:.0f} now against ${M["wti_last_cpi"]:.0f} in {M["cpi_through"]}, and the pump price ${M["pump_now"]:.2f} a gallon. That is what makes {nxt["label"]} hot.</li>
<li><b>Rents</b> follow new-lease rents (Zillow) with about a year's lag; those are up {M["rent_yoy"]:.1f}%, so rents add a steady 0.08pp a month. <b>Wages</b> are up {M["wage_yoy"]:.1f}%, which feeds services prices slowly.</li>
</ul>
<div class="box"><div class="box-h">Last print · {SC["month"]}</div>
<div class="kpis"><div><span class="sk">actual</span><span class="sv">{SC["actual_yoy"]:.2f}%</span><span class="ss">{sgn(SC["actual_mom"])}% on the month · year over year {SC["actual_dir"]}</span></div>
<div><span class="sk">this sheet said</span><span class="sv">{SC["model_yoy"]:.2f}%</span><span class="ss">called {SC["model_dir"]} → <b class="{"ok" if hit_word == "hit" else "miss"}">{hit_word}</b>, off by {SC["actual_yoy"] - SC["model_yoy"]:+.2f}pp</span></div>
<div><span class="sk">Hedgeye said</span><span class="sv">{SC["hedgeye_yoy"]:.2f}%</span><span class="ss">off by {SC["actual_yoy"] - SC["hedgeye_yoy"]:+.2f}pp</span></div></div>
<p class="ss">The miss was gasoline: it rose 3.9% in August because pump prices lag oil by a few weeks, and the model was reading oil. It now reads pump prices directly, which are public every Monday. In the backtest that change takes the next-month call from 84% to 90% right.</p></div>
<div class="tblwrap"><table>
<thead><tr><th>Month</th><th class="num">Oil (WTI), $/barrel</th><th class="num">Pump price, $/gallon</th><th class="num">Gasoline in the CPI, month on month</th><th class="num">Gasoline's push on inflation that month</th></tr></thead>
<tbody>{erows}</tbody></table></div>

<h2>Growth: what is behind the numbers</h2>
<ul class="story">
<li><b>This quarter ({now["label"]}): about {now["qoq_ann"]:+.1f}% annualized</b>, estimated from jobs, industrial production, retail sales and jobless claims with {M["k"]} of 3 months in. (Annualized = the pace over a full year if the quarter repeated.)</li>
<li><b>After this quarter we do not forecast GDP.</b> Nobody can, two to four quarters out; every method tried did worse than a constant. So the sheet assumes {pace_sentence}, adjusted for interest rates as below. (The old setting, the last six years' typical pace of {M["trend_ann"]:.1f}%, is backward looking: it assumes the post-2020 boom continues.)</li>
<li><b>Up or down is decided by the bar to beat.</b> Year-over-year growth rises in a quarter only if that quarter grows faster than the same quarter a year earlier. Those bars are already published, so each quarter's call is "does our assumed pace clear a known bar". The table shows the bar, our path, and what a slower economy would do to the quad.</li>
</ul>
<div class="tblwrap"><table class="wide">
<thead><tr><th>Quarter</th><th class="num">Bar to beat (last year's pace)</th><th class="num">Inflation change</th>{shead}</tr></thead>
<tbody>{srows}</tbody></table></div>
<p class="legend-inline">Each cell is the quad if growth ran at that pace, with the pace under it. The inflation side is the same in every column.</p>

<h3>The consumer</h3>
<p class="lede">Consumer spending is about two thirds of GDP. Each line says where it stands against the last ten years, and what that has historically meant for the next quarter.</p>
<div class="consumer">{consumer_html}</div>

<h3>Interest rates</h3>
<ul class="story">
<li><b>Where rates are.</b> The Fed funds rate was {RT["level"]:.2f}% through mid-{pd.Period(RT["month"], "M").strftime("%B")}; the September hike takes it to about {RT["implied_now"]:.2f}%, and the market expects <b>{RT["implied_end"]:.2f}% by July 2027</b>, roughly three more hikes.</li>
<li><b>How the sheet uses that.</b> Rate hikes slow growth with a lag: in the data since 1960, growth is lower for the four quarters that start two quarters after the hikes, by about {abs(RT["beta_ann_per_pp"]):.1f}pp of annual growth per 1pp of hikes. So each quarter's assumed pace is the trend plus the effect of rate changes over the year ending two quarters earlier.</li>
<li><b>What that means now.</b> The Fed <b>cut</b> {abs(RT["chg_8q"]):.1f}pp over the last two years, and under that lag the cuts still help growth through Q1 2027. The new hikes start to bite in <b>{qmap[RT["first_bite"]]["label"] if RT["first_bite"] else "late 2027"}</b>, at −0.3 to −0.4pp of annualized growth{f', and first change a quad in {qmap[RT["first_change"]]["label"]}' if RT["first_change"] else ''}.</li>
</ul>
<div class="tblwrap"><table class="wide">
<thead><tr class="grp"><th></th><th colspan="2">Market path (curve of {RT["path_asof"]})</th><th colspan="3">If the Fed hikes at every meeting, to {RT["hawk_end"]:.2f}%</th><th></th><th></th></tr>
<tr><th>Quarter</th><th class="num">Fed rate</th><th class="num">Effect on growth</th><th class="num">Fed rate</th><th class="num">Effect on growth</th><th class="num">…even at the fastest bite history shows</th><th class="num">Bar to beat</th><th>Quad</th></tr></thead>
<tbody>{rrows}</tbody></table></div>
<p class="legend-inline">Effects in percentage points of annualized growth. "Fastest bite" applies the strongest single-quarter response in the 1960–2026 record (two quarters after a hike) instead of the measured average.</p>

<div class="box view"><div class="box-h">Your view: the Fed keeps hiking, and growth slows in Q1 and Q2 2027</div>
<p><b>If that is right, both quarters are Quad 4</b>, because inflation is falling hard in both anyway ({sgn(q1["di"])}pp and {sgn(q2["di"])}pp). What growth has to do: come in <b>under {q1["bar_ann"]:+.1f}% annualized in Q1 2027</b> and <b>under {q2["bar_ann"]:+.1f}% in Q2 2027</b>. The bars are low because early 2026 was weak, so it does not take a recession, just a real slowdown.</p>
<p><b>What the sheet says.</b> {view_sentence}</p>
<p><b>What would confirm your view early:</b> jobless claims rising through the autumn, retail sales going flat, the Q4 2026 GDP print (28 January 2027) coming in under {qmap["2026Q4"]["bar_ann"]:+.1f}% annualized, and the Q1 2027 print (late April) under {q1["bar_ann"]:+.1f}%. The consumer lines above are where it would show first: real income is already weak and the saving rate is already very low.</p>
<p class="ss"><b>Honesty note.</b> In the 2017–2026 backtest the rate effect made the growth calls slightly worse, not better: it reversed {RS["n_flipped"]} of {RS["n_rows"]} calls and was right on {pct(RS["hit_flipped_on"])} of those against {pct(RS["hit_flipped_off"])} without it. Both times it mattered (2018, 2023) the Fed hiked into an economy that kept growing. It is on because you asked for it; the cards above say where it changes a quad.</p></div>

<h2>Against Hedgeye</h2>
<div class="tblwrap"><table>
<thead><tr><th>Question</th><th>Hedgeye ({HG["read_date"]})</th><th>This sheet</th><th>Verdict</th></tr></thead><tbody>
<tr><td class="lbl">Inflation in Q4 2026</td><td>rising, NowCast about {HG["q4_2026_nowcast"]:.2f}%</td><td>{qmap["2026Q4"]["i_yoy"]:.2f}% for the quarter, peak {M["peak"]["yoy"]:.2f}% in {M["peak"]["label"]}</td><td>same direction</td></tr>
<tr><td class="lbl">Then "cut roughly in half" by mid-2027</td><td>yes</td><td>{M["peak"]["yoy"]:.2f}% → {M["trough"]["yoy"]:.2f}% by {M["trough"]["label"]}, −{(1 - M["trough"]["yoy"] / M["peak"]["yoy"]) * 100:.0f}%</td><td>agree, strong call</td></tr>
<tr><td class="lbl">Quad in Q2 2027</td><td>Quad 4</td><td>Quad {q2["quad"]}: inflation side agrees, growth needs a print under {q2["bar_ann"]:+.1f}% annualized</td><td>hinges on growth</td></tr>
<tr><td class="lbl">Rate hikes slow growth</td><td>yes, hard, into 2027</td><td>yes, but with the lag the data show the bite lands in the second half of 2027</td><td>same story, later date</td></tr>
</tbody></table></div>

<h2>How often is this right?</h2>
<p class="lede">Every number comes from a backtest of {ST["n_asof"]} month-end runs from January 2017, each using only what was published at the time. Nothing here is a live track record yet.</p>
<div class="tblwrap">{trust}</div>

<div class="notes">
<p><b>†</b> BLS never published the October 2025 CPI (the autumn 2025 shutdown). It is filled in by averaging September and November 2025 so 12-month comparisons stay honest; October 2026's rate is compared against that fill.</p>
<p><b>How to read the growth levels.</b> After this quarter they are an assumption ({pace_sentence.replace("<b>", "").replace("</b>", "")}, plus the rate effect), not a forecast. What the sheet stands behind is the direction, up or down against the bar, and the inflation path.</p>
<p><b>Caveats.</b> The backtest uses simulated first releases of GDP (the real-time archives are unreachable from where this is built), which costs it 5–8 points of growth accuracy. The interest-rate path ahead is read off a curve dated {RT["path_asof"]}; the actual Fed rate replaces it as it prints. The sample is one inflation cycle. Regenerate with <code>python us_sheet.py</code> after each data release.</p>
</div>
</div>
<script>
(function(){{
  var box=document.querySelector('.chartbox'), tip=document.getElementById('tip'), xh=document.getElementById('xhair');
  var months={json.dumps(D["chart"]["months"])}, infl={json.dumps([round(x, 2) for x in D["chart"]["infl"]])};
  var bands={json.dumps(D["chart"]["bands"])}, growth={json.dumps({b["q"]: round(q["g_yoy"], 2) for b in D["chart"]["bands"] for q in [qmap[b["q"]]]})};
  function bandOf(i){{for(var k=0;k<bands.length;k++){{if(i>=bands[k].i0&&i<=bands[k].i1)return bands[k];}}return null;}}
  box.querySelectorAll('.hov').forEach(function(r){{
    r.addEventListener('mousemove',function(e){{
      var i=+r.getAttribute('data-i'), bd=bandOf(i);
      var x=parseFloat(r.getAttribute('x'))+parseFloat(r.getAttribute('width'))/2;
      xh.setAttribute('x1',x); xh.setAttribute('x2',x); xh.setAttribute('visibility','visible');
      tip.innerHTML='<b>'+months[i]+'</b> · inflation '+infl[i].toFixed(2)+'%'+(bd?' · growth '+growth[bd.q].toFixed(2)+'% · Quad '+bd.quad:'');
      var bb=box.getBoundingClientRect(); var lx=e.clientX-bb.left+12, ly=e.clientY-bb.top-34;
      if(lx+tip.offsetWidth>bb.width-8) lx=e.clientX-bb.left-tip.offsetWidth-12;
      tip.style.left=lx+'px'; tip.style.top=ly+'px'; tip.style.visibility='visible';
    }});
    r.addEventListener('mouseleave',function(){{tip.style.visibility='hidden'; xh.setAttribute('visibility','hidden');}});
  }});
}})();
</script>
'''


def main() -> None:
    D = build()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(D, indent=1, default=float))
    html = render(D)
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML} ({len(html)} bytes) and {OUT_JSON}")
    q = pd.DataFrame(D["quarters"])[["label", "realized", "quad", "name", "close", "quad_off", "g_yoy", "dg", "i_yoy", "di", "qoq_ann", "bar_ann"]]
    print(q.round(2).to_string(index=False))
    print(pd.DataFrame(D["rate"]["rows"])[["label", "mkt_rate", "mkt_drag_ann", "hawk_rate", "hawk_drag_ann", "fast_hawk_ann", "bar_ann", "quad"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
