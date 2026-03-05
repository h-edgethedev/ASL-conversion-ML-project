import queue
import time
import numpy as np
import streamlit as st
import cv2
import torch
from torch import nn

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from mediapipe.python.solutions import hands as mp_hands

SEQ_LEN = 10
FEAT_DIM = 63
MODEL_PATH = "models/lstm_model.pt"

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    model = LSTMClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes

def extract_features(results):
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)

    # normalize
    wrist = pts[0].copy()
    pts = pts - wrist
    ref = np.linalg.norm(pts[9])
    if ref < 1e-6:
        return None
    pts = pts / ref

    return pts.flatten()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)  # store latest frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # keep latest frame available
        if not self.frame_queue.full():
            self.frame_queue.put(img)
        return frame  # show raw frame in UI

def main():
    st.set_page_config(page_title="LSTM Sign → Text", page_icon="🤟")
    st.title("🤟 LSTM Sign → Text (Live Camera + Record)")
    st.caption("Camera stays on. Click Record to capture ~1s and predict hello/yes.")

    st.sidebar.header("Recording Settings")
    duration_sec = st.sidebar.slider("Record duration (seconds)", 1.0, 5.0, 2.0, 0.5)
    required_frames = st.sidebar.slider("Frames to collect (SEQ_LEN)", 10, 30, SEQ_LEN, 5)

    # (optional) update global SEQ_LEN behavior for this run
    seq_len = required_frames

    # (optional) update global SEQ_LEN behavior for this run
    seq_len = required_frames

    model, classes = load_model()

    ctx = webrtc_streamer(
        key="sign-lstm",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    record = st.button("🎥 Record 1 second & Predict")

    if record:
        if ctx.video_processor is None:
            st.error("Camera not ready yet. Wait a second and try again.")
            return

    # UI indicators
    status_box = st.empty()
    progress = st.progress(0)
    countdown_box = st.empty()

    status_box.info("🔴 REC • Recording started... keep your hand in frame")

    seq = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        start = time.time()
        while True:
            elapsed = time.time() - start
            remaining = max(0.0, duration_sec - elapsed)

            # update UI
            pct = min(1.0, elapsed / duration_sec) if duration_sec > 0 else 1.0
            progress.progress(int(pct * 100))
            countdown_box.write(f"⏱️ Recording… **{remaining:.1f}s** left | Frames: **{len(seq)}/{seq_len}**")

            if elapsed >= duration_sec:
                break

            try:
                frame_bgr = ctx.video_processor.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_bgr = cv2.flip(frame_bgr, 1)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feats = extract_features(results)
            if feats is not None:
                seq.append(feats)

            # if we already got enough frames, we can stop early
            if len(seq) >= seq_len:
                break

    progress.progress(100)
    status_box.success("✅ Recording complete")

    if len(seq) < seq_len:
        st.error(f"Not enough hand frames captured ({len(seq)}/{seq_len}). Try better lighting / keep hand steady.")
        return

    # build tensor
    X = np.stack(seq[:seq_len], axis=0)  # (T, 63)

    # If model was trained with SEQ_LEN=10, we must feed 10.
    # So we handle this safely:
    if X.shape[0] != SEQ_LEN:
        # Simple fix: if longer, take last 10 frames; if shorter, pad by repeating last frame
        if X.shape[0] > SEQ_LEN:
            X = X[-SEQ_LEN:]
        else:
            last = X[-1]
            pad = np.repeat(last[None, :], SEQ_LEN - X.shape[0], axis=0)
            X = np.vstack([X, pad])

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        conf = float(probs[pred_idx])

    st.success(f"Prediction: **{pred_label}**")
    st.write(f"Confidence: `{conf:.2f}`")

if __name__ == "__main__":
    main()