"""
Fire Detection
Real-time fire & smoke detection with a professional dashboard look.

Run:   streamlit run fire_detection_app.py
Needs: pip install streamlit ultralytics opencv-python numpy pandas requests
"""
import glob
import io
import os
import re
import sys
import threading
import time
import wave
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------- theme file
THEME = """[theme]
base = "light"
primaryColor = "#E8501A"
backgroundColor = "#F7F6F4"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0F172A"
"""
cfg = os.path.join(".streamlit", "config.toml")
try:
    current = open(cfg).read() if os.path.exists(cfg) else ""
    if current != THEME:
        os.makedirs(".streamlit", exist_ok=True)
        with open(cfg, "w") as f:
            f.write(THEME)
except OSError:
    pass  # theme is optional; the CSS below still styles the app

st.set_page_config(page_title="Fire Detection", page_icon="🔥", layout="wide")

# Streamlit Community Cloud runs apps from /mount/src; a server has no webcam.
ON_CLOUD = os.path.exists("/mount/src")

# Newer Streamlit uses width="stretch"; older uses use_container_width=True.
_ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
.stApp *:not([data-testid="stIconMaterial"]):not(.material-icons){font-family:'Inter','Segoe UI',system-ui,-apple-system,Roboto,Arial,sans-serif}
.stApp h1,.stApp h2,.stApp h3,.brand span,.hero-body h1{font-family:'Inter Tight','Inter','Segoe UI',system-ui,sans-serif !important}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none}
[data-testid="stExpandSidebarButton"],[data-testid="collapsedControl"]{color:#fff}
.stApp{background:#F7F6F4}
.block-container{max-width:none;padding:0 3.5rem 3rem}
section[data-testid="stSidebar"]{border-right:1px solid #E5E2DD}
section[data-testid="stSidebar"] h2{color:#6B6560;font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-top:.6rem;padding-bottom:4px;border-bottom:1px solid #E5E2DD}
.stTabs [data-baseweb="tab"]{font-weight:600}

/* ---------- hero ---------- */
.hero{margin:0 -3.5rem;padding:0 3.5rem 230px;min-height:640px;color:#fff;position:relative;overflow:hidden;
 background:radial-gradient(900px 520px at 78% 8%,rgba(232,80,26,.38),transparent 62%),
 radial-gradient(760px 520px at 8% 95%,rgba(150,45,40,.45),transparent 62%),
 linear-gradient(165deg,#1B0F1E 0%,#3A1B2C 48%,#5B2B2B 100%)}
.nav{display:flex;align-items:center;justify-content:space-between;padding:26px 0;gap:24px}
.brand{display:flex;align-items:center;gap:10px}
.brand span{font-weight:600;letter-spacing:.16em;font-size:1.02rem;color:#fff}
.menu{display:flex;gap:32px}
.menu a,.login{color:#fff !important;text-decoration:none !important;font-size:.76rem;font-weight:500;letter-spacing:.09em;text-transform:uppercase}
.menu a:hover{text-decoration:underline !important}
.navr{display:flex;align-items:center;gap:24px}
.btn{display:inline-block;background:#fff;color:#111 !important;padding:14px 26px;border-radius:3px;font-size:.76rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;text-decoration:none !important}
.btn:hover{background:#FDE8DC}
.btn.big{padding:18px 42px;font-size:.84rem}
.hero-body{text-align:center;padding-top:70px}
.stApp .hero-body h1{color:#fff;font-weight:500;font-size:4.3rem;line-height:1.03;letter-spacing:-.035em;margin:0 0 36px;padding:0}
.hero-sub{color:rgba(255,255,255,.86);margin-top:24px;font-size:.98rem}

/* ---------- stage (overlaps hero) ---------- */
.st-key-stage{margin-top:-200px;position:relative;z-index:3}
.modelline{color:rgba(255,255,255,.82);font-size:.85rem;padding-top:8px}
.modelline b{color:#fff}
.pill{display:inline-block;padding:5px 15px;border-radius:99px;font-size:.78rem;font-weight:600;background:rgba(255,255,255,.14);color:#fff;border:1px solid rgba(255,255,255,.28)}
.pill.live{background:#16A34A;border-color:#16A34A}
.pill.err{background:#DC2626;border-color:#DC2626}
.st-key-viewfinder{padding:14px;background:#0B0A10;border:1px solid #2A2430;border-radius:10px;min-height:320px;box-shadow:0 30px 60px rgba(10,5,15,.4)}
.idle{color:#A8A1AE;text-align:center;padding:110px 20px;font-size:1.05rem}
.stat,.gauge{background:#fff;border:1px solid #E8E4DF;border-radius:10px;padding:12px 16px;margin-bottom:10px;box-shadow:0 10px 24px rgba(20,10,20,.12)}
.stat span,.gauge-title{display:block;font-size:.72rem;font-weight:500;color:#7A736C;text-transform:uppercase;letter-spacing:.06em}
.stat b{font-family:'Inter Tight',sans-serif;font-size:1.8rem;line-height:1.2;color:#17121A;font-weight:600}
.gauge-level{font-family:'Inter Tight',sans-serif;font-size:1.7rem;font-weight:600;line-height:1.2}
.advice{font-size:.88rem;color:#5E5852;margin-top:2px}
.bar{height:8px;background:#ECE8E3;border-radius:6px;overflow:hidden;margin-top:8px}
.bar div{height:100%;border-radius:6px}
.chip{display:inline-block;margin:0 6px 6px 0;padding:3px 11px;border-radius:99px;background:#1B0F1E;color:#fff;font-size:.85rem;font-weight:500}
.chip i{font-style:normal;color:#FDBA74;margin-left:6px;font-weight:700}
.alarm{background:#DC2626;color:#fff;border-radius:10px;padding:14px 20px;margin-bottom:12px;font-weight:600;font-size:1.08rem;animation:pulse 1s infinite alternate}
.alarm small{display:block;font-weight:400;opacity:.92;margin-top:2px}
@keyframes pulse{from{box-shadow:0 0 0 0 rgba(220,38,38,.55)}to{box-shadow:0 0 18px 6px rgba(220,38,38,.35)}}
.armed{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-left:4px solid #22C55E;border-radius:10px;padding:10px 16px;margin-bottom:12px;color:#F1EEF3;font-size:.92rem}
.sos{background:#1B0F1E;color:#D9D2DC;border-radius:10px;padding:12px 16px;margin-bottom:10px;font-size:.88rem}
.sos b{color:#fff;font-size:1.1rem}

/* ---------- how it works ---------- */
.section-title{font-family:'Inter Tight',sans-serif !important;font-size:2rem;font-weight:600;letter-spacing:-.02em;color:#17121A;margin:46px 0 4px}
.section-sub{color:#6B6560;margin-bottom:18px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.step{background:#fff;border:1px solid #E8E4DF;border-radius:10px;padding:22px 22px 24px}
.step .num{font-family:'Inter Tight',sans-serif;color:#E8501A;font-weight:600;font-size:.85rem;letter-spacing:.1em}
.step h3{font-family:'Inter Tight',sans-serif !important;font-size:1.3rem;margin:6px 0 6px;color:#17121A;font-weight:600;padding:0}
.step p{color:#5E5852;margin:0;font-size:.95rem;line-height:1.55}
@media (max-width:900px){.steps{grid-template-columns:1fr}.menu{display:none}.stApp .hero-body h1{font-size:2.6rem}}
.footer{text-align:center;color:#8A837C;font-size:.85rem;margin-top:34px;border-top:1px solid #E5E2DD;padding-top:14px}
.st-key-alarm_audio{height:0;overflow:hidden;opacity:0}
</style>
""",
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("peak", Counter())
ss.setdefault("log", [])
ss.setdefault("alerts", [])
ss.setdefault("timeline", [])
ss.setdefault("siren_n", 0)
ss.setdefault("last_log", 0.0)
ss.setdefault("last_tl", 0.0)
ss.setdefault("dl_n", 0)


LEVELS = ["Safe", "Watch", "Danger", "Critical"]

ADVICE = {
    "Safe": "No fire detected. Keep exits clear.",
    "Watch": "Possible fire or smoke. Go and check the area now.",
    "Danger": "Fire likely. Raise the alarm, leave the area, call the fire service.",
    "Critical": "Large fire. Evacuate at once. Do not fight it. Call the fire service.",
}


@st.cache_resource
def load_model(path):
    return YOLO(path)


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file", "my_model.pt")
    try:
        model = load_model(model_path)
    except Exception as e:
        found = ", ".join(sorted(glob.glob("*.pt"))) or "none found in this folder"
        st.error(f"Can't load '{model_path}'.\n\nModel files here: {found}\n\n{e}")
        st.stop()

    st.header("Source")
    source = st.radio(
        "Input",
        ["Browser camera", "Webcam", "Image", "Video file"],
        index=0 if ON_CLOUD else 1,
        horizontal=True,
        label_visibility="collapsed",
        help="Browser camera works everywhere (online or local). Webcam only works when the app runs on your own computer.",
    )
    cam_index, res = 0, "1280x720"
    if source == "Webcam":
        cam_index = st.number_input("Camera number (0 = first camera)", 0, 10, 0)
        res = st.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)

    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
    iou = st.slider("Overlap filter (IoU)", 0.10, 0.95, 0.70, 0.05)
    imgsz = st.select_slider("Speed vs accuracy (image size)", [320, 480, 640, 960], value=640)
    all_names = list(model.names.values())
    chosen = st.multiselect("Only detect these classes", all_names, placeholder="All classes")
    class_ids = [k for k, v in model.names.items() if v in chosen] or None

    st.header("Display")
    show_labels = st.toggle("Show labels", True)
    show_conf = st.toggle("Show confidence", True)
    thickness = st.slider("Box thickness", 1, 8, 3)

    st.header("Alarm")
    alarm_on = st.toggle("Enable alarm", True)
    default_alarm = [n for n in all_names if any(k in n.lower() for k in ("fire", "smoke", "flame"))]
    alarm_classes = st.multiselect("Trigger on", all_names, default=default_alarm, placeholder="Any class")
    alarm_conf = st.slider("Alarm confidence", 0.05, 0.99, 0.60, 0.05)
    hold_secs = st.slider("Must be seen for (seconds)", 0.0, 5.0, 1.0, 0.5)
    cooldown = st.slider("Wait between alarms (seconds)", 2, 60, 10)
    min_level = st.select_slider("Sound alarm from threat level", LEVELS[1:], value="Danger")
    sound_on = st.toggle("Play siren", True)
    siren_style = st.selectbox("Siren style", ["Two-tone", "Rapid beep", "Rising sweep"])
    siren_vol = st.slider("Siren volume", 0.1, 1.0, 0.7, 0.1)
    auto_snap = st.toggle("Save a photo when the alarm fires", True)
    emergency_no = st.text_input("Emergency number to display", "112")
    tg_on = st.toggle("Send Telegram alert")
    tg_token = tg_chat = ""
    tg_min = "Danger"
    if tg_on:
        tg_min = st.select_slider("Telegram from threat level", LEVELS[1:], value="Danger")
        tg_token = st.text_input("Bot token", type="password")
        tg_chat = st.text_input("Chat ID")
        test_tg = st.button("Test Telegram", **STRETCH)
    else:
        test_tg = False
    test_sound = st.button("Test siren", **STRETCH)

    snap = record = False
    if source not in ("Image", "Browser camera"):
        st.header("Output")
        record = st.toggle("Record video to file")
        snap = st.button("Save snapshot", **STRETCH)

    if st.button("Reset session stats", **STRETCH):
        ss.peak, ss.log, ss.alerts, ss.timeline = Counter(), [], [], []

watch_set = set(alarm_classes)


# ------------------------------------------------------------------- helpers
def roman(n):
    if n <= 0:
        return "—"
    n = min(int(n), 3999)
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                 (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out += s
            n -= v
    return out


def detect(frame):
    r = model(frame, conf=conf, iou=iou, classes=class_ids, imgsz=imgsz, verbose=False)[0]
    img = r.plot(labels=show_labels, conf=show_conf, line_width=thickness)
    names = [model.names[int(c)] for c in r.boxes.cls.tolist()]
    return img, names, r.boxes.conf.tolist(), r.boxes.xyxy.tolist()


def analyse(names, confs, boxes, shape):
    """Return (threat score 0-100, % of frame covered by watched objects)."""
    area_total = float(shape[0] * shape[1]) or 1.0
    top, area = 0.0, 0.0
    for n, c, b in zip(names, confs, boxes):
        if watch_set and n not in watch_set:
            continue
        top = max(top, c)
        area += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    area_pct = min(100.0, area / area_total * 100)
    score = 0.0 if top == 0 else min(100.0, top * 70 + min(area_pct, 15) * 2)
    return score, area_pct


def level_of(score):
    if score < 1:
        return "Safe", "#16A34A"
    if score < 45:
        return "Watch", "#D97706"
    if score < 70:
        return "Danger", "#EA580C"
    return "Critical", "#DC2626"


def remember(names, confs, score):
    for k, v in Counter(names).items():
        ss.peak[k] = max(ss.peak[k], v)
    now = time.time()
    if names and now - ss.last_log > 0.5 and len(ss.log) < 5000:
        t = datetime.now().strftime("%H:%M:%S")
        ss.log += [{"time": t, "class": n, "confidence": round(c, 3)} for n, c in zip(names, confs)]
        ss.last_log = now
    if now - ss.last_tl > 0.5:
        ss.timeline.append(round(score, 1))
        ss.timeline = ss.timeline[-600:]
        ss.last_tl = now


def paint(fps, names, score, area_pct):
    lvl, color = level_of(score)
    chips = "".join(f'<span class="chip">{k}<i>{v}</i></span>' for k, v in Counter(names).items())
    fps_html = f'<div class="stat"><span>Frames per second</span><b>{fps:.0f}</b></div>' if fps else ""
    stats_slot.markdown(
        f'<div class="gauge"><div class="gauge-title">THREAT LEVEL</div>'
        f'<div class="gauge-level" style="color:{color}">{lvl}</div>'
        f'<div class="advice">{ADVICE[lvl]}</div>'
        f'<div class="bar"><div style="width:{score:.0f}%;background:{color}"></div></div></div>'
        + fps_html
        + f'<div class="stat"><span>Objects in view</span><b>{len(names)}</b></div>'
        + f'<div class="stat"><span>Frame covered by watched objects</span><b>{area_pct:.1f}%</b></div>'
        + (f"<div>{chips}</div>" if chips else ""),
        unsafe_allow_html=True,
    )


def draw_charts():
    if ss.peak:
        peak_slot.bar_chart(pd.Series(dict(ss.peak)), color="#16325C", height=170)
    if ss.timeline:
        line_slot.line_chart(pd.Series(ss.timeline[-300:]), color="#DC2626", height=150)


def idle(msg):
    frame_slot.markdown(f'<div class="idle">{msg}</div>', unsafe_allow_html=True)


def siren_wav(n):
    rate, secs = 22050, 2.0
    t = np.linspace(0, secs, int(rate * secs), endpoint=False)
    gate = np.ones_like(t)
    if siren_style == "Rapid beep":
        freq = np.full_like(t, 1000.0 + n % 7)
        gate = ((t * 8) % 1 < 0.5).astype(float)
    elif siren_style == "Rising sweep":
        freq = 500.0 + 700.0 * ((t * 1.5) % 1)
    else:
        freq = np.where((t * 3) % 1 < 0.5, 880 + n % 7, 660 + n % 7)
    pcm = (siren_vol * 0.9 * gate * np.sin(2 * np.pi * np.cumsum(freq) / rate) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_siren():
    ss.siren_n += 1
    audio_slot.audio(siren_wav(ss.siren_n), format="audio/wav", autoplay=True)


def send_telegram(token, chat, jpg, caption):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat, "caption": caption},
            files={"photo": ("alert.jpg", jpg)},
            timeout=10,
        )
    except Exception:
        pass


def fire_alarm(img, hits, lvl):
    labels = ", ".join(sorted({n for n, _ in hits}))
    top = max(c for _, c in hits)
    stamp = datetime.now()
    ss.alerts.append({"time": stamp.strftime("%H:%M:%S"), "detected": labels,
                      "confidence": round(top, 2), "threat": lvl})
    if sound_on:
        play_siren()
    ok, buf = cv2.imencode(".jpg", img)
    if auto_snap:
        os.makedirs("alerts", exist_ok=True)
        cv2.imwrite(f"alerts/alert_{stamp:%Y%m%d_%H%M%S}.jpg", img)
    if tg_on and tg_token and tg_chat and ok and LEVELS.index(lvl) >= LEVELS.index(tg_min):
        caption = f"🔥 {lvl} alert: {labels} ({top:.0%}) at {stamp:%H:%M:%S}"
        threading.Thread(target=send_telegram, args=(tg_token, tg_chat, buf.tobytes(), caption), daemon=True).start()


def armed_html():
    what = ", ".join(alarm_classes) or "any object"
    return f'<div class="armed">Monitoring active. Watching for: <b>{what}</b></div>'


def check_alarm(names, confs, img, state, hold, lvl):
    if not alarm_on:
        return
    if LEVELS.index(lvl) < LEVELS.index(min_level):
        if time.time() - state["seen"] > 1.5:
            state["since"] = None
        return
    hits = [(n, c) for n, c in zip(names, confs) if c >= alarm_conf and (not watch_set or n in watch_set)]
    now = time.time()
    if hits:
        state["seen"] = now
        state["since"] = state["since"] or now
        if now - state["since"] >= hold:
            labels = ", ".join(sorted({n for n, _ in hits}))
            if now - state["last"] >= cooldown:
                fire_alarm(img, hits, lvl)
                state["last"] = now
                state["banner"] = None
            if state["banner"] != "alarm":
                state["banner"] = "alarm"
                alert_slot.markdown(
                    f'<div class="alarm">FIRE ALARM: {labels} detected · {lvl}'
                    f'<small>{datetime.now():%H:%M:%S} — {ADVICE[lvl]} Call {emergency_no or "112"} / Fire 101</small></div>',
                    unsafe_allow_html=True,
                )
    elif now - state["seen"] > 1.5:
        state["since"] = None
        if state["banner"] != "armed":
            state["banner"] = "armed"
            alert_slot.markdown(armed_html(), unsafe_allow_html=True)


def render_tables(buttons):
    with log_slot.container():
        if ss.log:
            df = pd.DataFrame(ss.log)
            st.dataframe(df.tail(200), hide_index=True, **STRETCH)
            if buttons:
                ss.dl_n += 1
                st.download_button("Download log as CSV", df.to_csv(index=False), "detections.csv",
                                   "text/csv", key=f"dl_log_{ss.dl_n}")
        else:
            st.write("Detections will appear here once the model finds something.")
    with hist_slot.container():
        if ss.alerts:
            adf = pd.DataFrame(ss.alerts)
            st.dataframe(adf, hide_index=True, **STRETCH)
            if buttons:
                ss.dl_n += 1
                st.download_button("Download alert report (CSV)", adf.to_csv(index=False),
                                   "alert_report.csv", "text/csv", key=f"dl_alert_{ss.dl_n}")
        else:
            st.write("No alarms yet. They appear here when a trigger class is detected.")
    with gallery_slot.container():
        files = sorted(glob.glob(os.path.join("alerts", "*.jpg")))[-6:][::-1]
        if files:
            cols = st.columns(3)
            for i, fpath in enumerate(files):
                cols[i % 3].image(fpath, caption=os.path.basename(fpath), **STRETCH)
        else:
            st.write("Alert photos will be shown here.")


def tg_test():
    if not (tg_token and tg_chat):
        st.toast("Enter the bot token and chat ID first.")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                          data={"chat_id": tg_chat, "text": "🔥 Fire Detection: test message"}, timeout=8)
        st.toast("Telegram works ✅" if r.ok else f"Telegram error: {r.status_code}")
    except Exception as e:
        st.toast(f"Telegram failed: {e}")


# -------------------------------------------------------------------- layout
st.markdown(
    '<div class="hero"><div class="nav">'
    '<div class="brand"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 2c1 3 5 5 5 10a5 5 0 0 1-10 0c0-2 1-3 2-4 0 2 1 3 2 3 0-3-1-6 1-9z"/></svg>'
    '<span>FIRE DETECTION</span></div>'
    '<div class="menu"><a href="#live" target="_self">Live monitor</a><a href="#records" target="_self">Records</a>'
    '<a href="#how" target="_self">How it works</a></div>'
    f'<div class="navr"><span class="login">Emergency {emergency_no or "112"}</span>'
    '<a class="btn" href="#live" target="_self">Get started</a></div></div>'
    '<div class="hero-body"><h1>Catch fire early.<br>Act before it spreads.</h1>'
    '<a class="btn big" href="#live" target="_self">Start detection</a>'
    '<div class="hero-sub">Real-time fire and smoke detection on your own camera. Free to run, no cloud needed.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)
audio_slot = st.container(key="alarm_audio").empty()

with st.container(key="stage"):
    st.markdown('<div id="live"></div>', unsafe_allow_html=True)
    top_l, top_r = st.columns([5, 1])
    top_l.markdown(
        f'<div class="modelline">Model in service &nbsp;·&nbsp; <b>{os.path.basename(model_path)}</b></div>',
        unsafe_allow_html=True,
    )
    pill_slot = top_r.empty()
    alert_slot = st.empty()

    left, right = st.columns([3, 1.4])
    with left:
        with st.container(key="viewfinder"):
            frame_slot = st.empty()
    with right:
        stats_slot = st.empty()
        st.markdown(
            f'<div class="sos">Emergency numbers (India)<br><b>{emergency_no or "112"}</b> · Fire <b>101</b> · Ambulance <b>108</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Threat over time**")
        line_slot = st.empty()
        st.markdown("**Most seen at once**")
        peak_slot = st.empty()

st.markdown('<div id="records"></div>', unsafe_allow_html=True)
tab_log, tab_alerts, tab_gallery, tab_plan, tab_safety = st.tabs(
    ["Detection log", "Alert history", "Alert photos", "Alarm plan", "Safety measures"]
)
with tab_log:
    log_slot = st.empty()
with tab_alerts:
    hist_slot = st.empty()
with tab_gallery:
    gallery_slot = st.empty()
with tab_plan:
    st.markdown("**How the alarm works.** Detections move through four stages. The siren starts at your chosen level.")
    st.table(pd.DataFrame([
        {"Stage": "1 · Safe", "When": "Nothing watched is in view", "What happens": "System stays armed, green status"},
        {"Stage": "2 · Watch", "When": "Low score (small or uncertain flame/smoke)", "What happens": "Banner shows 'go and check'; no siren by default"},
        {"Stage": "3 · Danger", "When": "Medium score", "What happens": "Siren, photo saved, alert logged, Telegram if enabled"},
        {"Stage": "4 · Critical", "When": "High score (large or very certain fire)", "What happens": "Same as Danger, plus an evacuate message; siren repeats every cooldown"},
    ]))
    st.markdown(
        f"**Current settings:** alarm {'on' if alarm_on else 'off'} · watching "
        f"{', '.join(alarm_classes) or 'any class'} · confidence ≥ {alarm_conf:.0%} · seen for {hold_secs:g}s · "
        f"siren from **{min_level}** ({siren_style}, volume {siren_vol:.0%}) · repeats every {cooldown}s · "
        f"Telegram {'from ' + tg_min if tg_on else 'off'}"
    )
    st.caption("A fire detector can miss a fire. Never rely on it alone: keep smoke alarms and extinguishers in place.")
with tab_safety:
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            "**When the alarm sounds (RACE)**\n"
            "1. **Rescue**: help anyone in immediate danger, only if it is safe.\n"
            "2. **Alarm**: shout, trigger the building alarm, call the fire service.\n"
            "3. **Contain**: close doors and windows behind you to slow the spread.\n"
            "4. **Evacuate / Extinguish**: leave first. Fight a fire only if it is small and you have a clear exit.\n\n"
            "**Using an extinguisher (PASS)**\n"
            "Pull the pin · Aim at the base · Squeeze the handle · Sweep side to side."
        )
    with s2:
        st.markdown(
            "**Do**\n"
            "- Stay low under smoke and feel doors before opening them.\n"
            "- Use stairs, never lifts.\n"
            "- Meet at the assembly point and count everyone.\n"
            "- Call 112 or Fire 101 from outside.\n\n"
            "**Don't**\n"
            "- Don't go back inside for belongings.\n"
            "- Don't use water on oil or electrical fires.\n"
            "- Don't block exits or prop fire doors open."
        )
    st.markdown("**Readiness checklist**")
    c1, c2 = st.columns(2)
    for i, item in enumerate([
        "Exits and corridors are clear", "Extinguisher checked and in date", "Smoke alarms tested this month",
        "Camera sees the high-risk area", "Everyone knows the assembly point", "Emergency numbers displayed",
    ]):
        (c1 if i % 2 == 0 else c2).checkbox(item, key=f"chk_{i}")

pill_slot.markdown('<span class="pill">Idle</span>', unsafe_allow_html=True)
draw_charts()
paint(0, [], 0.0, 0.0)

alarm_state = {"seen": 0.0, "since": None, "last": 0.0, "banner": "armed" if alarm_on else None}
if alarm_on:
    alert_slot.markdown(armed_html(), unsafe_allow_html=True)
if test_sound:
    play_siren()
if test_tg:
    tg_test()

# --------------------------------------------------------------------- modes
live = False
if source in ("Image", "Browser camera"):
    if source == "Image":
        up = left.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    else:
        up = left.camera_input("Take a photo to check for fire", help="Allow camera access in your browser when asked.")
    if up:
        frame = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("That file could not be read as an image.")
        else:
            img, names, confs, boxes = detect(frame)
            score, area_pct = analyse(names, confs, boxes, frame.shape)
            lvl, _ = level_of(score)
            remember(names, confs, score)
            frame_slot.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **STRETCH)
            check_alarm(names, confs, img, {"seen": 0.0, "since": None, "last": 0.0, "banner": None}, 0, lvl)
            paint(0, names, score, area_pct)
            draw_charts()
            ok, buf = cv2.imencode(".png", img)
            if ok:
                left.download_button("Download result", buf.tobytes(), "detected.png", "image/png")
    else:
        idle("Upload an image to see detections." if source == "Image"
             else "Press <b>Take photo</b> below. The photo is checked for fire straight away.")
else:
    cap = None
    if source == "Webcam":
        if st.sidebar.toggle("Camera on", key="run_cam"):
            w, h = map(int, res.split("x"))
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(cam_index), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    else:
        up = left.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
        if up:
            os.makedirs("uploads_tmp", exist_ok=True)
            path = os.path.join("uploads_tmp", os.path.basename(up.name))
            with open(path, "wb") as f:
                f.write(up.getbuffer())
            if st.sidebar.toggle("Process video", key="run_vid"):
                cap = cv2.VideoCapture(path)

    if cap is None:
        idle("Camera is off. Switch on <b>Camera on</b> in the sidebar." if source == "Webcam"
             else "Upload a video, then switch on <b>Process video</b>.")
        render_tables(True)
    elif not cap.isOpened():
        cap.release()
        pill_slot.markdown('<span class="pill err">No camera</span>', unsafe_allow_html=True)
        idle("No camera found here.")
        if source == "Webcam":
            st.warning(
                "No webcam could be opened. If you are using the online version, the server has no camera: "
                "choose **Browser camera**, **Image** or **Video file** in the sidebar. "
                "On your own PC, try another camera number and close other apps that use the webcam."
            )
        else:
            st.error("That video could not be opened. Try an .mp4 file.")
        render_tables(True)
    else:
        live = True
        pill_slot.markdown('<span class="pill live">● Live</span>', unsafe_allow_html=True)
        writer, n, prev, snap_pending, fails = None, 0, time.time(), snap, 0
        render_tables(False)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if source == "Webcam" and fails < 30:
                        time.sleep(0.05)
                        continue
                    break
                fails = 0
                img, names, confs, boxes = detect(frame)
                score, area_pct = analyse(names, confs, boxes, frame.shape)
                lvl, _ = level_of(score)
                remember(names, confs, score)
                check_alarm(names, confs, img, alarm_state, hold_secs, lvl)
                frame_slot.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **STRETCH)

                if record:
                    if writer is None:
                        os.makedirs("recordings", exist_ok=True)
                        out = f"recordings/rec_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20
                        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), min(fps_in, 30),
                                                 (img.shape[1], img.shape[0]))
                    writer.write(img)
                if snap_pending:
                    os.makedirs("snapshots", exist_ok=True)
                    cv2.imwrite(f"snapshots/snap_{datetime.now():%Y%m%d_%H%M%S}.jpg", img)
                    st.toast("Snapshot saved to the snapshots folder")
                    snap_pending = False

                n += 1
                if n % 5 == 0:
                    now = time.time()
                    paint(5 / max(now - prev, 1e-6), names, score, area_pct)
                    prev = now
                if n % 15 == 0:
                    draw_charts()
                if n % 30 == 0:
                    render_tables(False)
        except Exception as e:
            st.error(f"Detection stopped: {e}")
        finally:
            cap.release()
            if writer:
                writer.release()
        idle("Finished. Switch the toggle off and on to run again.")
        pill_slot.markdown('<span class="pill">Stopped</span>', unsafe_allow_html=True)
        draw_charts()
        render_tables(True)

if source in ("Image", "Browser camera"):
    render_tables(True)

st.markdown(
    '<div id="how"></div><div class="section-title">How it works</div>'
    '<div class="section-sub">Three steps from camera to alarm.</div>'
    '<div class="steps">'
    '<div class="step"><div class="num">01</div><h3>Detect</h3><p>Your YOLO model scans every frame from the webcam, a video or an image and marks fire and smoke.</p></div>'
    '<div class="step"><div class="num">02</div><h3>Assess</h3><p>Confidence and the share of the frame covered are combined into a Safe, Watch, Danger or Critical level.</p></div>'
    '<div class="step"><div class="num">03</div><h3>Alert</h3><p>The siren sounds, a photo is saved, the alert is logged and a Telegram message can be sent to your phone.</p></div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="footer">Fire Detection · Real-time monitoring · In an emergency call 112 / Fire 101</div>', unsafe_allow_html=True)
