import time

import cv2
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Detector", layout="wide")
st.title("YOLO Object Detection")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model path", "my_model.pt")
cam_index = st.sidebar.number_input("Camera index (0 = usb0)", 0, 10, 0)
thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
res = st.sidebar.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)
run = st.sidebar.toggle("Start camera")


@st.cache_resource
def load_model(path):
    return YOLO(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Could not load model '{model_path}': {e}")
    st.stop()

frame_slot = st.empty()
col1, col2 = st.columns(2)
fps_slot = col1.empty()
count_slot = col2.empty()

if run:
    w, h = map(int, res.split("x"))
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        st.error("Could not open camera. Try a different index or close other apps using it.")
        st.stop()

    prev = time.time()
    while run:
        ok, frame = cap.read()
        if not ok:
            st.warning("Lost camera feed.")
            break

        results = model(frame, conf=thresh, verbose=False)[0]
        annotated = results.plot()  # BGR image with boxes drawn

        now = time.time()
        fps = 1 / max(now - prev, 1e-6)
        prev = now

        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        fps_slot.metric("FPS", f"{fps:.1f}")
        count_slot.metric("Objects detected", len(results.boxes))

    cap.release()
else:
    st.info("Turn on **Start camera** in the sidebar to begin.")
