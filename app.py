"""TraceQ prototype: Quality Intelligence decision support on UCI SECOM.

AI ranks batches and may raise them to "Unusual" at most. Only a human reviewer, with an
evidence note, can move a batch to "Under investigation", "Confirmed defect" or "Cleared".
Preprocessing and models mirror reference/secom_analysis.py exactly.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
QUEUE_SLOTS, AUDIT_SLOTS, SEED = 20, 2, 7

STATUS_COLORS = {
    "Normal": "#8D99AE",
    "Unusual": "#F5B31B",
    "Under investigation": "#7DBBDD",
    "Confirmed defect": "#C8553D",
    "Cleared": "#2A9D8F",
}


# ----------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Scoring the test period (fits 3 models, ~20 s once)...")
def build():
    X = pd.read_csv(DATA / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"S{i}" for i in range(X.shape[1])]
    lab = pd.read_csv(DATA / "secom_labels.data", sep=r"\s+", header=None, names=["y", "ts"], quotechar='"')
    lab["ts"] = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    y = (lab["y"] == 1).astype(int).values
    order = np.argsort(lab["ts"].values, kind="stable")
    X, y, ts = X.iloc[order].reset_index(drop=True), y[order], lab["ts"].iloc[order].reset_index(drop=True)

    n_tr = int(len(X) * 0.7)
    Xtr, Xte, ytr, yte = X.iloc[:n_tr], X.iloc[n_tr:], y[:n_tr], y[n_tr:]
    keep = [c for c in X.columns if Xtr[c].isna().mean() <= 0.5 and Xtr[c].nunique(dropna=True) > 1]
    med = Xtr[keep].median()
    Atr, Ate = Xtr[keep].fillna(med), Xte[keep].fillna(med)
    sc = StandardScaler().fit(Atr)
    Ztr, Zte = np.clip(sc.transform(Atr), -10, 10), np.clip(sc.transform(Ate), -10, 10)

    def rank(s):
        return pd.Series(s).rank(pct=True).values

    iso = IsolationForest(n_estimators=500, random_state=7).fit(Ztr[ytr == 0])
    s_iso = -iso.score_samples(Zte)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                                random_state=7, n_jobs=-1).fit(Ztr, ytr)
    s_rf = rf.predict_proba(Zte)[:, 1]
    lr = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", class_weight="balanced").fit(Ztr, ytr)
    s_lr = lr.predict_proba(Zte)[:, 1]
    s_tq = (rank(s_iso) + rank(s_rf) + rank(s_lr)) / 3

    # Why flagged: z vs passing training runs, on the scaled data
    pass_mu, pass_sd = Ztr[ytr == 0].mean(0), Ztr[ytr == 0].std(0) + 1e-9
    Z = (Zte - pass_mu) / pass_sd
    top3 = np.argsort(-np.abs(Z), axis=1)[:, :3]
    drivers = [[(keep[j], float(Z[i, j])) for j in top3[i]] for i in range(len(Zte))]

    test = pd.DataFrame({
        "batch": [f"B-{order[n_tr + i]}" for i in range(len(Zte))],  # B- + original row index in secom.data
        "ts": ts.iloc[n_tr:].values,
        "score": s_tq,
        "pct": rank(s_tq) * 100,
        "rf": s_rf,
        "label": np.where(yte == 1, "fail", "pass"),
    })
    test["drivers"] = drivers

    o = np.argsort(-s_tq)
    top = list(o[:QUEUE_SLOTS])
    rest = np.setdiff1d(np.arange(len(test)), top)
    audit = list(np.random.default_rng(SEED).choice(rest, AUDIT_SLOTS, replace=False))

    # Evaluation strip: the static Random Forest from analysis/secom_extra.py (400 trees, pandas-std
    # scaling), so the app shows the same capture number as the deck.
    mu_e, sd_e = Atr.mean(), Atr.std() + 1e-9
    to_z = lambda A: np.clip(((A - mu_e) / sd_e).values, -10, 10)
    rf_eval = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                     random_state=7, n_jobs=-1).fit(to_z(Atr), ytr)
    s_eval = rf_eval.predict_proba(to_z(Ate))[:, 1]
    budget = int(round(0.2 * len(yte)))
    rf_capture = yte[np.argsort(-s_eval)[:budget]].sum() / yte.sum()
    top_pass = int((yte[o[:QUEUE_SLOTS]] == 0).sum())
    return test, top, audit, {"rf_capture": rf_capture, "top_pass": top_pass, "budget": budget,
                              "n_test": len(yte), "n_fail": int(yte.sum())}


# ----------------------------------------------------------------------------- helpers
def chip(status):
    c = STATUS_COLORS[status]
    return (f"<span style='background:{c}22;color:{c};border:1px solid {c};border-radius:999px;"
            f"padding:1px 10px;font-size:0.78rem;white-space:nowrap'>{status}</span>")


def why_chart(drivers, height=210):
    d = drivers[::-1]  # largest at top
    names = [s for s, _ in d]
    vals = [z for _, z in d]
    fig = go.Figure(go.Bar(
        x=[abs(v) for v in vals], y=names, orientation="h",
        marker_color=["#F5B31B" if v > 0 else "#7DBBDD" for v in vals],
        text=[f"{v:+.1f}σ" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(size=15, color="#E6E6E6"),
    ))
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="|σ| vs. passing runs in training data", gridcolor="#333", zeroline=False),
        yaxis=dict(tickfont=dict(size=14)), showlegend=False,
    )
    return fig


def set_status(batch, new, note):
    old = st.session_state.status[batch]
    st.session_state.status[batch] = new
    st.session_state.audit.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "batch": batch,
                                   "from → to": f"{old} → {new}", "note": note.strip()})
    st.session_state.note_key += 1


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="TraceQ", page_icon="🔎", layout="wide")
test, top, audit, ev = build()
queue = [(i, "Ranked") for i in top] + [(i, "Random audit") for i in audit]

ss = st.session_state
if "status" not in ss:
    # The AI may raise a batch to "Unusual" at most. Random audit slots start at "Normal".
    ss.status = {test.batch[i]: ("Unusual" if kind == "Ranked" else "Normal") for i, kind in queue}
    ss.audit, ss.note_key = [], 0
    ss.sel = test.batch[top[0]]

st.markdown("""<style>
.block-container{padding-top:2.2rem;padding-bottom:1rem}
.qrow{display:flex;align-items:center;gap:10px;font-size:0.9rem;line-height:1.9}
.mono{font-family:ui-monospace,Consolas,monospace}
.muted{color:#8D99AE}
.note{background:#1d2333;border-left:3px solid #F5B31B;padding:8px 12px;border-radius:4px;font-size:0.9rem}
.strip{background:#161b26;border:1px solid #2b3245;border-radius:8px;padding:10px 16px;font-size:1.0rem}
</style>""", unsafe_allow_html=True)

# Compact layout for the slide card screenshot: /?view=strip
if st.query_params.get("view") == "strip":
    st.markdown("<style>header,[data-testid='stToolbar']{display:none!important}"
                ".block-container{padding:0.6rem 1.2rem!important}</style>", unsafe_allow_html=True)
    b = test.iloc[top[0]]
    st.markdown("<div style='font-size:1.05rem'><b>TraceQ</b> · Line 3 (UCI SECOM test period) · "
                "Today's queue: 20 inspection slots</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        rows = "".join(
            f"<div class='qrow'><span class='mono' style='width:62px'>{test.batch[i]}</span>"
            f"<span class='muted' style='width:120px'>{pd.Timestamp(test.ts[i]):%Y-%m-%d %H:%M}</span>"
            f"<span style='width:62px'>P{test.pct[i]:.1f}</span>{chip(ss.status[test.batch[i]])}</div>"
            for i in top[:5])
        st.markdown(f"<div class='muted' style='font-size:0.8rem'>RANKED QUEUE (top 5 of 20)</div>{rows}",
                    unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='muted' style='font-size:0.8rem'>WHY FLAGGED · {b.batch} · "
                    f"vs. passing runs in training data</div>", unsafe_allow_html=True)
        st.plotly_chart(why_chart(b.drivers, height=190), use_container_width=True,
                        config={"displayModeBar": False})
    st.stop()

st.markdown("### TraceQ · Line 3 (UCI SECOM test period) · Today's queue: 20 inspection slots")
st.caption("Decision support only. The model can raise a batch to **Unusual**; only a reviewer with an "
           "evidence note can move it further up the Evidence Ladder: Normal → Unusual → Under investigation "
           "→ Confirmed defect / Cleared.")

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("#### Ranked queue")
    for n, (i, kind) in enumerate(queue):
        if n == QUEUE_SLOTS:
            st.markdown("<div class='muted' style='margin-top:6px'>Exploration slots</div>", unsafe_allow_html=True)
        bid = test.batch[i]
        rank_lbl = f"#{n + 1}" if kind == "Ranked" else "Random audit"
        c1, c2 = st.columns([5, 1])
        c1.markdown(
            f"<div class='qrow'><span class='muted' style='width:92px'>{rank_lbl}</span>"
            f"<span class='mono' style='width:62px'><b>{bid}</b></span>"
            f"<span class='muted' style='width:122px'>{pd.Timestamp(test.ts[i]):%Y-%m-%d %H:%M}</span>"
            f"<span style='width:58px'>P{test.pct[i]:.1f}</span>{chip(ss.status[bid])}</div>",
            unsafe_allow_html=True)
        if c2.button("Open", key=f"open_{bid}", type="primary" if ss.sel == bid else "secondary",
                     use_container_width=True):
            ss.sel = bid
            st.rerun()

with right:
    b = test[test.batch == ss.sel].iloc[0]
    status = ss.status[b.batch]
    st.markdown(f"#### {b.batch} &nbsp; {chip(status)}", unsafe_allow_html=True)
    st.caption(f"Run time {pd.Timestamp(b.ts):%Y-%m-%d %H:%M} · TraceQ rank percentile P{b.pct:.1f} "
               f"(average of Isolation Forest, Random Forest and L1 logistic percentile ranks)")

    st.markdown("**Why flagged** <span class='muted'>· top 3 signals vs. passing runs in training data</span>",
                unsafe_allow_html=True)
    st.plotly_chart(why_chart(b.drivers), use_container_width=True, config={"displayModeBar": False})
    st.markdown("".join(f"- Signal **{s}** is **{z:+.1f}σ** vs passing runs\n" for s, z in b.drivers))
    st.markdown("<div class='note'>Unusual is not the same as defective. "
                "Realistic AUC on this data is about 0.62.</div>", unsafe_allow_html=True)

    st.markdown("**Reviewer actions**")
    note = st.text_input("Evidence note (required)", key=f"note_{ss.note_key}_{b.batch}",
                         placeholder="What did you check, and what did you find? Press Enter to apply.")
    has_note = bool(note.strip())
    a1, a2, a3 = st.columns(3)
    can_start = status in ("Normal", "Unusual")
    can_close = status == "Under investigation"
    if a1.button("Start investigation", disabled=not (has_note and can_start), use_container_width=True):
        set_status(b.batch, "Under investigation", note); st.rerun()
    if a2.button("Confirm defect", disabled=not (has_note and can_close), use_container_width=True):
        set_status(b.batch, "Confirmed defect", note); st.rerun()
    if a3.button("Clear", disabled=not (has_note and can_close), use_container_width=True):
        set_status(b.batch, "Cleared", note); st.rerun()
    if not has_note:
        st.caption("Buttons unlock once an evidence note is entered.")
    elif not can_start and not can_close:
        st.caption("This batch already has a reviewer decision.")

    if st.toggle("Reveal final test result (demo only)", key="reveal"):
        c = "#2A9D8F" if b.label == "pass" else "#C8553D"
        st.markdown(f"Final SECOM test result: <b style='color:{c}'>{b.label.upper()}</b>"
                    + (" · an unusual batch can still pass." if b.label == "pass" else ""),
                    unsafe_allow_html=True)

    st.markdown("**Audit log**")
    if ss.audit:
        st.dataframe(pd.DataFrame(ss.audit[::-1]), hide_index=True, use_container_width=True)
    else:
        st.caption("No reviewer changes yet.")

st.markdown("---")
st.markdown(
    f"<div class='strip'><b>Evaluation (test period, {ev['n_test']} runs, {ev['n_fail']} fails):</b> "
    f"Inspecting the top 20% catches <b>{ev['rf_capture'] * 100:.0f}%</b> of fails vs ~20% random "
    f"<span class='muted'>(Random Forest score, {ev['budget']} runs)</span> · "
    f"<b>{ev['top_pass']}/{QUEUE_SLOTS}</b> top flags passed final test</div>",
    unsafe_allow_html=True)
st.caption("Data: UCI SECOM (McCann & Johnston, 2008), CC BY 4.0. No operator, shift or personnel data is used.")
