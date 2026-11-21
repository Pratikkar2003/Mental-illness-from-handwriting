"""
app.py
------
Simple upload UI.

Run with:
    streamlit run app.py

Upload a handwriting image, get a predicted emotion + probability breakdown.
"""

import os
import sys
import numpy as np
import cv2
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from feature_extraction import extract_features  # noqa: E402
from predict import load_model  # noqa: E402
import torch  # noqa: E402

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

st.set_page_config(page_title="Handwriting Emotion Prototype", page_icon=" ")
st.title(" Handwriting Emotion Detection Prototype")

st.warning(
    "**Research prototype, not a validated emotional assessment.** "
    
)

checkpoint_ok = os.path.exists(os.path.join(CHECKPOINT_DIR, "model.pt")) and os.path.exists(
    os.path.join(CHECKPOINT_DIR, "labels.json")
)

if not checkpoint_ok:
    st.error(
        "No trained model found yet. Run `python src/generate_demo_data.py` and then "
        "`python src/train.py` first (see README.md)."
    )
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes = load_model(CHECKPOINT_DIR, device)

uploaded = st.file_uploader("Upload a handwriting image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded sample", use_container_width=True)

    feats = extract_features(image)
    if feats.seq_mask.sum() == 0:
        st.error("No handwriting/ink detected in this image. Try a clearer photo or scan.")
    else:
        point_seq = torch.from_numpy(feats.point_sequence).unsqueeze(0).to(device)
        seq_mask = torch.from_numpy(feats.seq_mask).unsqueeze(0).to(device)
        global_feats = torch.from_numpy(feats.global_features).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(point_seq, seq_mask, global_feats)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        ranked = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])
        top_class, top_conf = ranked[0]

        st.subheader(f"Predicted: **{top_class}**  ({top_conf:.1%} confidence)")
        st.bar_chart({c: p for c, p in ranked})
