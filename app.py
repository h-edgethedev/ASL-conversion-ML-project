import os
import time
import joblib
import numpy as np
import streamlit as st
import mediapipe as mp

MODEL_PATH = r"models\sign_model.joblib"

# IMPORTANT: Use the internal path since your mediapipe install doesn't expose mp.solutions
from mediapipe.python.solutions import hands as mp_hands

def extract_features(img_rgb):
    """
    Takes an RGB image (numpy array).
    Returns 63 features or None if no hand found.
    """
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    ) as hands:
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)

        # Normalize: move wrist to origin and scale by wrist->middle_mcp distance
        wrist = pts[0].copy()
        pts = pts - wrist
        ref = np.linalg.norm(pts[9])
        if ref < 1e-6:
            return None
        pts = pts / ref

        return pts.flatten()

def main():
    st.set_page_config(page_title="Sign → Text MVP", page_icon="🤟")
    st.title("🤟 Sign → Text (Hello vs Yes)")
    st.caption("Show a sign to the camera. This prototype predicts one of your trained labels.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train it first: `python train_model.py`")
        st.stop()

    model = joblib.load(MODEL_PATH)

    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.01)
    cooldown_ms = st.sidebar.slider("Cooldown (ms)", 0, 2000, 300, 50)

    cam = st.camera_input("Take a photo with your sign in frame")

    if "last_time" not in st.session_state:
        st.session_state.last_time = 0.0

    if cam is None:
        st.info("Use the camera above. Take a photo while holding the sign steady.")
        return

    now = time.time() * 1000
    if now - st.session_state.last_time < cooldown_ms:
        st.warning("Too fast—try again in a moment.")
        return

    from PIL import Image
    img = Image.open(cam).convert("RGB")
    img_rgb = np.array(img)

    feats = extract_features(img_rgb)
    if feats is None:
        st.error("No hand detected. Try better lighting and keep your hand fully in frame.")
        return

    # Predict + probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([feats])[0]
        classes = model.classes_
        best = int(np.argmax(probs))
        pred_label = str(classes[best])
        conf = float(probs[best])
    else:
        pred_label = str(model.predict([feats])[0])
        conf = 1.0

    if conf >= threshold:
        st.success(f"Prediction: **{pred_label}**")
    else:
        st.warning(f"Low confidence: **{pred_label}**")

    st.write(f"Confidence: `{conf:.2f}`")
    st.session_state.last_time = now

    st.divider()
    st.write("Model labels:", list(model.classes_))

if __name__ == "__main__":
    main()