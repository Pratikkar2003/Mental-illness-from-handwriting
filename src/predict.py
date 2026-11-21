"""
predict.py
----------
Usage:
    python src/predict.py --image path/to/handwriting.png

Loads the trained checkpoint and prints the predicted class plus the full
probability distribution.
"""

import os
import json
import argparse
import torch
import numpy as np

from feature_extraction import extract_features_from_path
from model import HandwritingEmotionTransformer


def load_model(checkpoint_dir, device):
    with open(os.path.join(checkpoint_dir, "labels.json")) as f:
        classes = json.load(f)
    model = HandwritingEmotionTransformer(num_classes=len(classes)).to(device)
    state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def predict(image_path: str, checkpoint_dir: str, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(checkpoint_dir, device)

    feats = extract_features_from_path(image_path)
    point_seq = torch.from_numpy(feats.point_sequence).unsqueeze(0).to(device)
    seq_mask = torch.from_numpy(feats.seq_mask).unsqueeze(0).to(device)
    global_feats = torch.from_numpy(feats.global_features).unsqueeze(0).to(device)

    if feats.seq_mask.sum() == 0:
        return {"error": "No handwriting/ink detected in this image."}

    with torch.no_grad():
        logits = model(point_seq, seq_mask, global_feats)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    ranked = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])
    return {
        "predicted_class": ranked[0][0],
        "confidence": ranked[0][1],
        "all_probabilities": {c: p for c, p in ranked},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "checkpoints"),
    )
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint_dir)
    if "error" in result:
        print(result["error"])
        return

    print(f"\nPredicted emotion: {result['predicted_class']}  (confidence {result['confidence']:.1%})")
    print("\nFull distribution:")
    for cls, p in result["all_probabilities"].items():
        print(f"  {cls:>10s}: {p:.1%}")
    print(
        "\nNote: this result is only as trustworthy as the data the model was "
        "trained on. See README.md before treating this as a real emotional "
        "assessment."
    )


if __name__ == "__main__":
    main()
