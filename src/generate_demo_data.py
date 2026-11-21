"""
generate_demo_data.py
----------------------
!!! THIS DATA IS SYNTHETIC AND FAKE !!!

This script procedurally draws squiggly "handwriting-like" line images with
different geometric parameters (slant, thickness, jitter, spacing) per class
label. It exists ONLY so the training/inference pipeline has something to
run on out of the box, so you can confirm the code works end-to-end on your
machine before you plug in real data.

The class names below (happy / sad / stressed / calm) are PLACEHOLDERS.
The synthetic images have no connection whatsoever to real human emotion or
real handwriting. A model trained only on this data tells you nothing about
real emotion recognition - it only proves the pipeline runs.

To get a model whose accuracy means anything, replace data/train and
data/val with real images, each placed in a folder named after its true
label, collected and labeled the way the paper describes (e.g. handwriting
samples paired with a validated instrument like the DASS questionnaire, and
informed consent from participants).
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

# purely synthetic geometric "personality" per placeholder class
CLASS_PARAMS = {
    "calm":     dict(slant=0,   jitter=1.0, thickness=2, spacing=18, wave_freq=0.15),
    "happy":    dict(slant=8,   jitter=2.0, thickness=3, spacing=14, wave_freq=0.35),
    "sad":      dict(slant=-10, jitter=0.5, thickness=1, spacing=10, wave_freq=0.08),
    "stressed": dict(slant=4,   jitter=4.5, thickness=2, spacing=22, wave_freq=0.55),
}

IMG_W, IMG_H = 600, 200


def draw_synthetic_sample(params: dict, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("L", (IMG_W, IMG_H), color=255)
    draw = ImageDraw.Draw(img)

    n_words = rng.integers(3, 6)
    x = 20
    baseline = IMG_H // 2
    for w in range(n_words):
        n_letters = rng.integers(3, 7)
        for l in range(n_letters):
            letter_h = rng.integers(20, 40)
            letter_w = rng.integers(10, 22)
            slant_rad = np.radians(params["slant"] + rng.normal(0, 2))
            n_pts = 12
            ys = np.linspace(-letter_h / 2, letter_h / 2, n_pts)
            wave = params["jitter"] * np.sin(ys * params["wave_freq"] + rng.normal(0, 1))
            xs = x + ys * np.tan(slant_rad) + wave + rng.normal(0, params["jitter"], n_pts)
            pts = list(zip(xs, baseline + ys))
            width = max(1, int(round(params["thickness"] + rng.normal(0, 0.4))))
            draw.line(pts, fill=0, width=width, joint="curve")
            x += letter_w + rng.normal(0, 1.5)
        x += params["spacing"]
        if x > IMG_W - 40:
            break
    return img


def generate(n_train_per_class=60, n_val_per_class=15, seed=0):
    random.seed(seed)
    for split, n in [("train", n_train_per_class), ("val", n_val_per_class)]:
        for cls, params in CLASS_PARAMS.items():
            out_dir = os.path.join(OUT_ROOT, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                img = draw_synthetic_sample(params, seed=hash((split, cls, i)) % (2**31))
                img.save(os.path.join(out_dir, f"{cls}_{i:03d}.png"))
    print(f"Synthetic demo dataset written under: {os.path.abspath(OUT_ROOT)}")
    print("Reminder: this data is fake and for pipeline-testing only.")


if __name__ == "__main__":
    generate()
