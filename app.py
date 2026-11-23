"""
Crowd Safety AI - live dashboard (Streamlit)

Run:
    python -m streamlit run app.py

Sources: a video file, a webcam, or a camera stream (RTSP / HTTP, for example a
phone running an IP-camera app). While footage is analysed the dashboard updates
live: frame, status sign, counts, crowd-size chart, zone heat and an alert log.
When it ends (or you press Stop) you can review it: scrub the footage, narrow the
time window and change the alert limits. Counts people only, no face recognition.
"""
import os
import tempfile
import time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Crowd Safety AI", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ tokens
INK = "#101C2B"
SLATE = "#5A6775"
CONCRETE = "#E4E8EA"
PANEL = "#D6DCDF"
LINE = "#C3CBD0"
LEVELS = ["Safe", "Watch", "Danger"]
LEVEL_COLOR = {"Safe": "#12805C", "Watch": "#F2B01E", "Danger": "#C8102E"}
LEVEL_TEXT = {"Safe": "#FFFFFF", "Watch": INK, "Danger": "#FFFFFF"}
LIVE_SAFETY_SECONDS = 1800  # live sources stop by themselves after 30 minutes


def hex_bgr(h):
    return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))


BGR = {k: hex_bgr(v) for k, v in LEVEL_COLOR.items()}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');

.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp li, .stApp h1, .stApp h2, .stApp h3 {
  font-family: 'Archivo', system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
}
.stApp { background: #E4E8EA; color: #101C2B; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1160px; padding-top: 1.6rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: #D6DCDF; }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

.brand { font-stretch: 125%; font-weight: 800; font-size: 1.05rem; letter-spacing: -.01em; margin: 0; }
.brandnote { color: #5A6775; font-size: .85rem; margin: .15rem 0 0; }
.masthead { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap; margin-bottom: 2.6rem; }

.title { font-stretch: 125%; font-weight: 900; font-size: clamp(2rem, 4.6vw, 3.4rem);
         line-height: 1.02; letter-spacing: -.025em; margin: 0 0 .9rem; max-width: 15ch; }
.lede { color: #5A6775; font-size: 1.08rem; line-height: 1.55; max-width: 56ch; margin: 0 0 2rem; }

.sec { font-stretch: 112%; font-weight: 800; font-size: 1.35rem; letter-spacing: -.01em; margin: 3rem 0 .25rem; }
.secnote { color: #5A6775; font-size: .95rem; margin: 0 0 1rem; max-width: 62ch; }

.verdict { font-stretch: 125%; font-weight: 900; font-size: clamp(1.7rem, 3.6vw, 2.6rem);
           line-height: 1.08; letter-spacing: -.02em; margin: 0 0 .5rem; max-width: 22ch; }
.verdictsub { color: #5A6775; font-size: 1.05rem; margin: 0 0 1.6rem; max-width: 60ch; }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1.4rem 2rem; margin: .4rem 0 .6rem; }
.facts b { display: block; font-stretch: 112%; font-weight: 800; font-size: 2.3rem; line-height: 1; letter-spacing: -.02em; }
.facts span { display: block; color: #5A6775; font-size: .88rem; line-height: 1.35; margin-top: .35rem; }

.sign { border-radius: 4px; padding: 1.3rem 1.4rem 1.5rem; min-height: 100%; }
.sign .word { font-stretch: 125%; font-weight: 900; font-size: 2.5rem; line-height: 1; letter-spacing: -.02em; }
.sign .num { font-stretch: 112%; font-weight: 800; font-size: 5.4rem; line-height: 1; letter-spacing: -.04em; margin-top: 1.4rem; }
.sign .cap { font-size: .95rem; margin-top: .3rem; opacity: .92; }
.sign .zone { font-size: .95rem; margin-top: 1.1rem; padding-top: .8rem; border-top: 1px solid currentColor;
              border-top-color: rgba(127,127,127,.55); }

.livelog { margin: 0; padding: 0; list-style: none; font-size: .95rem; line-height: 1.7; }
.livelog li { padding: .1rem 0; }
.livelog .d { display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .55rem; }
.livenow { display: inline-block; width: .6rem; height: .6rem; border-radius: 50%; background: #C8102E; margin-right: .5rem; }

.stButton > button, .stDownloadButton > button {
  border-radius: 4px; border: 0; background: #101C2B; color: #FFFFFF; font-weight: 600; padding: .6rem 1.2rem;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: #24364D; color: #FFFFFF; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid #F2B01E; outline-offset: 2px; }
[data-testid="stFileUploaderDropzone"] { background: #F4F6F7; border: 2px dashed #9AA7B0; border-radius: 4px; }
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .88rem; color: #101C2B; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


# ----------------------------------------------------------------- helpers
def fmt_t(s):
    s = max(float(s), 0.0)
    return f"{int(s // 60)}:{int(s % 60):02d}"


def risk_arrays(G, total_limit, cell_limit):
    """G has shape (frames, rows, cols). Returns total, peak, score, level index."""
    total = G.sum(axis=(1, 2))
    peak = G.max(axis=(1, 2))
    score = np.maximum(total / max(total_limit, 1), peak / max(cell_limit, 1))
    idx = np.where(score >= 1, 2, np.where(score >= 0.6, 1, 0))
    return total, peak, score, idx


def frame_df(times, total, peak, idx):
    """One row per frame. Each row ends where the next one starts."""
    times = np.asarray(times, dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    dt = max(dt, 1e-3)
    ends = np.append(times[1:], times[-1] + dt)
    return pd.DataFrame({"second": times, "end": ends, "people": np.asarray(total).astype(int),
                         "busiest": np.asarray(peak).astype(int), "level": [LEVELS[int(k)] for k in idx]})


def thin(df, n=600):
    step = max(1, int(np.ceil(len(df) / n)))
    return df.iloc[::step]


def detect(model, frame, rows, cols, conf, imgsz):
    h, w = frame.shape[:2]
    res = model(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
    grid = np.zeros((rows, cols), dtype=np.float32)
    points = []
    for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        grid[min(int(cy // (h / rows)), rows - 1), min(int(cx // (w / cols)), cols - 1)] += 1
        points.append((int(cx), int(cy)))
    return grid, points


def _text(img, s, org, scale, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(frame, grid, points, cell_limit, total_limit):
    """Draw the zone heat, people markers and a status bar on a BGR frame."""
    rows, cols = grid.shape
    h, w = frame.shape[:2]
    total, peak = int(grid.sum()), int(grid.max())
    level = LEVELS[int(risk_arrays(grid[None], total_limit, cell_limit)[3][0])]

    n = np.clip(grid / max(cell_limit, 1), 0, 1).astype(np.float32)
    n = np.clip(cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)[..., None]
    amber = np.array(BGR["Watch"], np.float32)
    red = np.array(BGR["Danger"], np.float32)
    tint = amber * (1 - n) + red * n
    alpha = np.where(n > 0.02, 0.14 + 0.48 * n, 0.0)
    out = (frame.astype(np.float32) * (1 - alpha) + tint * alpha).astype(np.uint8)

    cw, ch = w / cols, h / rows
    for r in range(1, rows):
        cv2.line(out, (0, int(r * ch)), (w, int(r * ch)), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        cv2.line(out, (int(c * cw), 0), (int(c * cw), h), (255, 255, 255), 1, cv2.LINE_AA)

    rad = max(2, w // 320)
    for x, y in points:
        cv2.circle(out, (x, y), rad + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), rad, (255, 255, 255), -1, cv2.LINE_AA)

    fs = max(0.45, w / 1500)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] > 0:
                _text(out, str(int(grid[r, c])), (int(c * cw) + 6, int(r * ch) + int(24 * fs * 1.4)), fs)

    bh = max(30, h // 16)
    cv2.rectangle(out, (0, 0), (w, bh), (29, 24, 16), -1)
    label = level.upper()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.05, 2)
    cv2.rectangle(out, (0, 0), (tw + 24, bh), BGR[level], -1)
    txt_col = (29, 24, 16) if level == "Watch" else (255, 255, 255)
    cv2.putText(out, label, (12, (bh + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.05, txt_col, 2, cv2.LINE_AA)
    cv2.putText(out, f"{total} people   busiest zone {peak}", (tw + 40, (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_events(df, kinds):
    cols = ["From", "To", "Length (s)", "Most people", "Busiest zone"]
    flag = df["level"].isin(kinds)
    if not flag.any():
        return pd.DataFrame(columns=cols)
    grp = (flag != flag.shift()).cumsum()
    rows = []
    for _, g in df[flag].groupby(grp[flag]):
        rows.append({
            "From": fmt_t(g["second"].iloc[0]), "To": fmt_t(g["end"].iloc[-1]),
            "Length (s)": round(float(g["end"].iloc[-1] - g["second"].iloc[0]), 1),
            "Most people": int(g["people"].max()), "Busiest zone": int(g["busiest"].max()),
        })
    return pd.DataFrame(rows, columns=cols)


def summarize(df):
    ev = make_events(df, ["Danger"])
    n = len(ev)
    risky = float((df["level"] != "Safe").mean())
    pk = df.loc[df["people"].idxmax()]
    if n:
        head = f"Danger threshold crossed {n} time{'s' if n != 1 else ''}."
    elif (df["level"] == "Watch").any():
        head = "Reached Watch level, never Danger."
    else:
        head = "No risk detected in this footage."
    sub = (f"The most people at once was {int(pk['people'])}, at {fmt_t(pk['second'])}. "
           f"{risky * 100:.0f}% of the selected time was Watch or Danger.")
    return head, sub, ev


def fact(value, label):
    return f"<div><b>{value}</b><span>{label}</span></div>"


def facts_html(items):
    return '<div class="facts">' + "".join(fact(v, l) for v, l in items) + "</div>"


def zone_line(grid):
    r, c = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return f"Busiest zone: row {r + 1}, column {c + 1}, {int(grid.max())} people"


def sign_html(level, number, caption, zone):
    return (f'<div class="sign" style="background:{LEVEL_COLOR[level]};color:{LEVEL_TEXT[level]}">'
            f'<div class="word">{level}</div><div class="num">{number}</div>'
            f'<div class="cap">{caption}</div><div class="zone">{zone}</div></div>')


def log_html(entries):
    if not entries:
        return '<p class="secnote">No alerts yet.</p>'
    items = "".join(f'<li><span class="d" style="background:{LEVEL_COLOR[lv]}"></span>{txt}</li>'
                    for lv, txt in entries[-6:][::-1])
    return f'<ul class="livelog">{items}</ul>'


# ------------------------------------------------------------------ charts
def style(ch):
    return (ch.configure_view(strokeWidth=0)
            .configure_axis(labelColor=SLATE, titleColor=SLATE, gridColor=LINE, domainColor=SLATE,
                            tickColor=SLATE, labelFont="Archivo", titleFont="Archivo",
                            labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure(background="transparent"))


def x_scale_for(df):
    return alt.Scale(domain=[float(df["second"].min()), float(df["end"].max())], nice=False)


def people_chart(df, total_limit, mark=None, height=230):
    d = thin(df)
    area = alt.Chart(d).mark_area(fillOpacity=0.14, color=INK, line={"color": INK, "strokeWidth": 1.6}).encode(
        x=alt.X("second:Q", scale=x_scale_for(df), title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("people:Q", title="People", axis=alt.Axis(minExtent=48)),
        tooltip=[alt.Tooltip("second:Q", title="Second", format=".1f"), alt.Tooltip("people:Q", title="People"),
                 alt.Tooltip("level:N", title="Level")])
    limit_rule = alt.Chart(pd.DataFrame({"y": [float(total_limit)]})).mark_rule(
        color=LEVEL_COLOR["Danger"], strokeDash=[6, 4], size=2).encode(y="y:Q")
    ch = area + limit_rule
    if mark is not None:
        ch = ch + alt.Chart(pd.DataFrame({"x": [float(mark)]})).mark_rule(color=INK, size=2).encode(x="x:Q")
    return ch.properties(height=height)


def ribbon_chart(df, mark=None):
    d = thin(df).assign(lane="Risk")
    ribbon = alt.Chart(d).mark_rect().encode(
        x=alt.X("second:Q", scale=x_scale_for(df), title="Seconds"),
        x2="end:Q",
        y=alt.Y("lane:N", title=None, axis=alt.Axis(minExtent=48, ticks=False, domain=False)),
        color=alt.Color("level:N", scale=alt.Scale(domain=LEVELS, range=[LEVEL_COLOR[x] for x in LEVELS]),
                        legend=alt.Legend(title=None, orient="top", symbolType="square", labelFontSize=12)),
        tooltip=[alt.Tooltip("second:Q", title="Second", format=".1f"), alt.Tooltip("level:N", title="Level")])
    ch = ribbon
    if mark is not None:
        ch = ch + alt.Chart(pd.DataFrame({"x": [float(mark)]})).mark_rule(color=INK, size=2).encode(x="x:Q")
    return ch.properties(height=34)


def zone_chart(zone, decimals=0, height=300):
    rr, cc = np.meshgrid(range(zone.shape[0]), range(zone.shape[1]), indexing="ij")
    zdf = pd.DataFrame({"row": rr.ravel() + 1, "col": cc.ravel() + 1, "people": zone.ravel().round(1)})
    zmax = max(float(zdf["people"].max()), 1.0)
    base = alt.Chart(zdf).encode(x=alt.X("col:O", title="Column", axis=alt.Axis(labelAngle=0)),
                                 y=alt.Y("row:O", title="Row", sort="ascending", axis=alt.Axis(labelAngle=0)))
    rect = base.mark_rect(stroke=CONCRETE, strokeWidth=3).encode(
        color=alt.Color("people:Q", title="People", scale=alt.Scale(
            domain=[0, zmax / 2, zmax], range=["#F4F6F7", LEVEL_COLOR["Watch"], LEVEL_COLOR["Danger"]])),
        tooltip=["row", "col", "people"])
    text = base.mark_text(fontSize=14, fontWeight="bold").encode(
        text=alt.Text("people:Q", format=f".{decimals}f"),
        color=alt.condition(alt.datum.people > zmax * 0.7, alt.value("#FFFFFF"), alt.value(INK)))
    return (rect + text).properties(height=height)


def show_chart(ch, where=st):
    try:
        where.altair_chart(style(ch), theme=None, width="stretch")
    except Exception:
        where.altair_chart(style(ch), theme=None, use_container_width=True)


def show_img(a, where=st):
    try:
        where.image(a, width="stretch")
    except Exception:
        where.image(a, use_container_width=True)


# ----------------------------------------------------------- live analysis
def open_capture(source, path_or_index):
    if source == "Webcam":
        cap = cv2.VideoCapture(int(path_or_index), cv2.CAP_DSHOW)  # fast backend on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(path_or_index))
        return cap
    return cv2.VideoCapture(path_or_index)


def analyse(cap, name, live, model, p):
    """Run the detector over a capture while painting a live dashboard.

    Results are written into st.session_state['res'] as they come in, so they are
    still there if the user presses Stop (which re-runs the script).
    """
    rows, cols, cell_limit, total_limit = p["rows"], p["cols"], p["cell_limit"], p["total_limit"]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps < 1 or fps > 120:
        fps = 25.0
    in_video = 0 if live else (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9)
    n_total = 10 ** 9 if live else max(1, min(in_video, int(p["max_sec"] * fps)))
    sample_every = 1 if live else max(1, n_total // 120)

    store = {"G": [], "times": [], "samples": {}, "video": None, "name": name, "live": live, "cut": False}
    st.session_state["res"] = store
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "annotated.mp4")
    writer = None

    box = st.empty()
    with box.container():
        st.markdown('<div class="sec" style="margin-top:2rem">'
                    + ('<span class="livenow"></span>Live' if live else "Analysing") + "</div>",
                    unsafe_allow_html=True)
        if live:
            st.button("Stop live monitoring", type="primary")
            bar = None
        else:
            bar = st.progress(0.0)
        status = st.empty()
        c1, c2 = st.columns([3, 1.15])
        frame_ph, sign_ph = c1.empty(), c2.empty()
        facts_ph = st.empty()
        chart_ph, ribbon_ph = st.empty(), st.empty()
        z1, z2 = st.columns([1.3, 1])
        zone_ph = z1.empty()
        with z2:
            st.markdown("**Alerts**")
            log_ph = st.empty()

    cnt = {"sum": 0.0, "peak": 0, "danger": 0, "events": 0, "prev": "Safe"}
    log = []
    grid, points = np.zeros((rows, cols), np.float32), []
    t0 = time.time()
    last_ui = 0.0
    last_sample = -1e9
    i = 0
    t_i = 0.0
    while i < n_total:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time() - t0
        if live and now > LIVE_SAFETY_SECONDS:
            break
        h, w = frame.shape[:2]
        if writer is None and not live:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if i % p["skip"] == 0:
            grid, points = detect(model, frame, rows, cols, p["conf"], p["imgsz"])
        t_i = now if live else i / fps
        store["G"].append(grid.copy())
        store["times"].append(t_i)

        total_i, peak_i = int(grid.sum()), int(grid.max())
        level_i = LEVELS[int(risk_arrays(grid[None], total_limit, cell_limit)[3][0])]
        cnt["sum"] += total_i
        cnt["peak"] = max(cnt["peak"], total_i)
        cnt["danger"] += level_i == "Danger"
        if level_i == "Danger" and cnt["prev"] != "Danger":
            cnt["events"] += 1
            log.append(("Danger", f"{fmt_t(t_i)}  Danger started, {total_i} people"))
        elif level_i != "Danger" and cnt["prev"] == "Danger":
            log.append((level_i, f"{fmt_t(t_i)}  Back to {level_i}"))
        cnt["prev"] = level_i

        shown = render(frame, grid, points, cell_limit, total_limit)
        if writer is not None:
            writer.write(shown)

        take = (now - last_sample >= 2.0) if live else (i % sample_every == 0)
        if take:
            last_sample = now
            small = frame if w <= 640 else cv2.resize(frame, (640, int(h * 640 / w)))
            sc = small.shape[1] / w
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
            store["samples"][i] = (buf.tobytes(), [(int(x * sc), int(y * sc)) for x, y in points])
            if len(store["samples"]) > 200:
                store["samples"].pop(next(iter(store["samples"])))

        if time.time() - last_ui > 0.4:
            last_ui = time.time()
            prev = shown if w <= 720 else cv2.resize(shown, (720, int(h * 720 / w)))
            show_img(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), frame_ph)
            sign_ph.markdown(sign_html(level_i, total_i, "people in view now" if live
                                       else f"people in view at {fmt_t(t_i)}", zone_line(grid)),
                             unsafe_allow_html=True)
            n_seen = len(store["G"])
            facts_ph.markdown(facts_html([
                (cnt["peak"], "most people so far"), (f"{cnt['sum'] / n_seen:.0f}", "people on average"),
                (cnt["events"], "danger alerts so far"),
                (f"{100 * cnt['danger'] / n_seen:.0f}%", "of the time at Danger")]), unsafe_allow_html=True)
            tail = min(n_seen, 4000)
            tt = np.array(store["times"][-tail:])
            GG = np.array(store["G"][-tail:])
            keep = tt >= tt[-1] - 90
            tt, GG = tt[keep], GG[keep]
            tot, pk, _, lv = risk_arrays(GG, total_limit, cell_limit)
            if len(tt) > 1:
                dfw = frame_df(tt, tot, pk, lv)
                show_chart(people_chart(dfw, total_limit, height=200), chart_ph)
                show_chart(ribbon_chart(dfw), ribbon_ph)
            show_chart(zone_chart(grid, 0, 230), zone_ph)
            log_ph.markdown(log_html(log), unsafe_allow_html=True)
            status.caption(f"Live for {fmt_t(now)}. Showing the last 90 seconds." if live
                           else f"Analysing frame {i + 1} of {n_total}")
            if bar is not None:
                bar.progress(min((i + 1) / n_total, 1.0))
        i += 1

    cap.release()
    if writer is not None:
        writer.release()
        with open(out_path, "rb") as f:
            store["video"] = f.read()
    store["cut"] = (not live) and i >= n_total and i < in_video
    box.empty()
    return store


# ---------------------------------------------------------------- sidebar
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Detector**")
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0,
                              help="Larger models find more people but run slower. Use the smallest for live cameras.")
    imgsz = st.select_slider("Detection size", options=[640, 960, 1280], value=640,
                             help="Larger sizes help find small, far-away people.")
    conf = st.slider("Confidence", 0.05, 0.60, 0.20, 0.05,
                     help="Lower finds more people and adds more false detections.")
    skip = st.slider("Detect every N frames", 1, 10, 3, help="Higher is faster. Frames in between reuse the last count.")
    max_sec = st.number_input("Video files: analyse the first (seconds)", 5, 600, 60,
                              help="Longer files are cut here. Live sources run until you press Stop.")
    rows = st.slider("Zone rows", 2, 8, 4)
    cols = st.slider("Zone columns", 2, 10, 6)
    st.markdown("**Alert limits**")
    cell_limit = st.number_input("Danger when one zone holds", 1, 100, 5)
    total_limit = st.number_input("Danger when the scene holds", 1, 1000, 50)
    st.caption("Watch starts at 60% of either limit. After the analysis, changing a limit updates the "
               "results straight away. Nothing is analysed again.")

# ------------------------------------------------------------------- page
st.markdown(
    '<div class="masthead"><p class="brand">Crowd Safety AI</p>'
    '<p class="brandnote">Counts people, not faces.</p></div>'
    '<h1 class="title">Find the moment a crowd becomes unsafe.</h1>'
    '<p class="lede">Upload footage or connect a camera. The dashboard counts people in every zone of the frame, '
    "updates as it watches, and marks each moment as Safe, Watch or Danger.</p>",
    unsafe_allow_html=True,
)

source = st.radio("Source", ["Video file", "Webcam", "Camera stream"], horizontal=True)
params = {"rows": rows, "cols": cols, "cell_limit": cell_limit, "total_limit": total_limit,
          "conf": conf, "imgsz": imgsz, "skip": skip, "max_sec": max_sec}
cap, cap_name, live = None, "", source != "Video file"

if source == "Video file":
    uploaded = st.file_uploader("Crowd footage", type=["mp4", "mov", "avi", "mkv"],
                                help="Footage shot from a high angle works best.")
    if uploaded is not None:
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.getbuffer())
        left, right = st.columns([3, 1])
        with left:
            st.video(in_path)
        with right:
            st.caption(f"{uploaded.name}, {uploaded.size / 1e6:.1f} MB")
            if st.button("Analyse footage", type="primary"):
                cap, cap_name = open_capture(source, in_path), uploaded.name
elif source == "Webcam":
    cam_no = st.number_input("Camera number", 0, 9, 0, help="0 is the built-in webcam on most laptops.")
    st.caption("The webcam is read by the computer that runs this app. On a hosted Space there is no camera, "
               "so use Webcam only when you run the app on your own computer.")
    if st.button("Start live monitoring", type="primary"):
        cap, cap_name = open_capture(source, cam_no), f"Webcam {int(cam_no)}"
else:
    url = st.text_input("Stream address", placeholder="rtsp://...  or  http://192.168.1.5:8080/video")
    st.caption("Any camera that exposes an RTSP or HTTP video address works. To use a phone as a camera, install an "
               "IP camera app on it, join the same Wi-Fi as this computer and paste the video address it shows.")
    if st.button("Start live monitoring", type="primary"):
        if url.strip():
            cap, cap_name = open_capture(source, url.strip()), "Camera stream"
        else:
            st.error("Paste the stream address first.")

if cap is not None:
    if not cap.isOpened():
        st.error("Could not open this source. Check the file, the camera number or the stream address.")
        st.stop()
    with st.spinner("Loading the detector. The first run downloads the model."):
        model = YOLO(model_name)
    analyse(cap, cap_name, live, model, params)

# ------------------------------------------------------------ review data
res = st.session_state.get("res")
if res is None or len(res["G"]) == 0:
    st.stop()

G = np.stack(res["G"])
times = np.array(res["times"], dtype=float)
F = len(G)
total, peak, score, lvl = risk_arrays(G, total_limit, cell_limit)
full = frame_df(times, total, peak, lvl)
duration = float(full["end"].iloc[-1])

st.markdown('<div class="sec">Time window</div>'
            '<p class="secnote">Narrow the window to study one part of the footage. '
            "Every figure below follows it.</p>", unsafe_allow_html=True)
dmax = round(duration, 1)
if dmax > 0.5:
    lo, hi = st.slider("Seconds", 0.0, dmax, (0.0, dmax), step=0.1)
else:
    lo, hi = 0.0, dmax
mask = (times >= lo) & (times <= hi)
if not mask.any():
    mask[:] = True
df = full[mask].reset_index(drop=True)

head, sub, danger_events = summarize(df)
st.markdown(f'<div class="sec" style="margin-top:2.4rem">What happened</div>'
            f'<p class="verdict">{head}</p><p class="verdictsub">{sub}</p>', unsafe_allow_html=True)
share = float((df["level"] == "Danger").mean()) * 100
st.markdown(facts_html([
    (int(df["people"].max()), f"people at the peak, limit {int(total_limit)}"),
    (f"{df['people'].mean():.0f}", "people on average"),
    (int(df["busiest"].max()), f"in the busiest zone, limit {int(cell_limit)}"),
    (f"{share:.0f}%", "of the time at Danger")]), unsafe_allow_html=True)
if res.get("cut"):
    st.caption("The footage was cut at the analysis limit set in the sidebar.")

# ------------------------------------------------------------------ scrub
st.markdown('<div class="sec">Review the footage</div>'
            '<p class="secnote">Drag the slider to step through the footage. The sign beside it '
            "shows how risky that moment is with the current limits.</p>", unsafe_allow_html=True)
if hi - lo > 0.2:
    t = st.slider("Time (seconds)", float(lo), float(hi), float(lo), step=0.1, key="scrub")
else:
    t = lo
keys = np.array(sorted(res["samples"].keys()))
want = int(np.argmin(np.abs(times - t)))
k = int(keys[np.argmin(np.abs(keys - want))])
jpg, pts = res["samples"][k]
frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
grid_k = G[min(k, F - 1)]
view = render(frame, grid_k, pts, cell_limit, total_limit)
tk, pk_, sk, lk = risk_arrays(grid_k[None], total_limit, cell_limit)
level_k = LEVELS[int(lk[0])]
t_k = float(times[min(k, F - 1)])

c1, c2 = st.columns([3, 1.15])
with c1:
    show_img(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
with c2:
    st.markdown(sign_html(level_k, int(tk[0]), f"people in view at {fmt_t(t_k)}", zone_line(grid_k)),
                unsafe_allow_html=True)

# ----------------------------------------------------------------- charts
st.markdown('<div class="sec">Crowd size over time</div>'
            '<p class="secnote">The dashed line is your limit for the whole scene. '
            "The strip below shows the risk level at every moment.</p>", unsafe_allow_html=True)
show_chart(people_chart(df, total_limit, mark=t_k))
show_chart(ribbon_chart(df, mark=t_k))

# ------------------------------------------------------------------ zones
st.markdown('<div class="sec">Where it gets crowded</div>'
            '<p class="secnote">Each square is one zone of the frame, as you see it in the footage.</p>',
            unsafe_allow_html=True)
mode = st.radio("Show", ["Average in the time window", "Peak in the time window", "The moment I scrubbed to"],
                horizontal=True)
Gw = G[mask]
zone = Gw.mean(axis=0) if mode.startswith("Average") else Gw.max(axis=0) if mode.startswith("Peak") else grid_k
show_chart(zone_chart(zone, 1 if mode.startswith("Average") else 0))

# ----------------------------------------------------------------- events
st.markdown('<div class="sec">Alert events</div>'
            '<p class="secnote">Each row is one stretch of time when the crowd stayed at that level.</p>',
            unsafe_allow_html=True)
kind = st.radio("Include", ["Danger only", "Watch and Danger"], horizontal=True)
ev = make_events(df, ["Danger"] if kind == "Danger only" else ["Watch", "Danger"])
if len(ev):
    st.dataframe(ev, hide_index=True)
else:
    st.caption("No events at this level. Lower the alert limits in the sidebar to see how sensitive it is.")

# ----------------------------------------------------------------- export
st.markdown('<div class="sec">Export</div>', unsafe_allow_html=True)
d1, d2, _ = st.columns([1, 1, 2])
if res.get("video"):
    d1.download_button("Download annotated video", res["video"], file_name="crowd_annotated.mp4", mime="video/mp4")
d2.download_button("Download data as CSV", df.drop(columns=["end"]).to_csv(index=False),
                   file_name="crowd_data.csv", mime="text/csv")
if res.get("video"):
    st.caption("The annotated video uses the alert limits from the moment you ran the analysis.")
