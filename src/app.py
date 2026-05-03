"""Gradio web demo for the Vehicle Damage Detector.

Mirrors the app currently deployed at
https://huggingface.co/spaces/rwandashi/vehicle-damage-detector — runs locally
with ``python -m src.app`` and opens at http://127.0.0.1:7860.

Use ``--share`` to also expose a temporary public URL (handy for showing the
demo from a Colab or remote box).
"""

from __future__ import annotations

import argparse

import gradio as gr
from PIL import Image
from ultralytics import YOLO

from src.predict import DEFAULT_HF_FILENAME, DEFAULT_HF_REPO, resolve_weights


def build_demo(model: YOLO) -> gr.Interface:
    def detect_damage(image):
        results = model.predict(image, conf=0.25)
        annotated = results[0].plot()

        detections = results[0].boxes
        if len(detections) == 0:
            summary = "No damage detected."
        else:
            lines = [f"Found {len(detections)} damage area(s):"]
            for box in detections:
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)
                lines.append(f"- {class_name}: {confidence:.0%} confidence")
            summary = "\n".join(lines)

        return Image.fromarray(annotated), summary

    return gr.Interface(
        fn=detect_damage,
        inputs=gr.Image(type="pil", label="Upload Car Photo"),
        outputs=[
            gr.Image(type="pil", label="Detected Damage"),
            gr.Textbox(label="Detection Summary"),
        ],
        title="Vehicle Damage Detector",
        description=(
            "Upload a photo of a car to automatically detect and classify damage. "
            "Powered by a YOLOv8m model fine-tuned on 6,839 images across 23 damage classes."
        ),
        allow_flagging="never",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None,
                        help="Local .pt path. Defaults to pulling from HF Hub.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--share", action="store_true",
                        help="Also create a temporary public Gradio URL.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = resolve_weights(args.weights, args.hf_repo, args.hf_filename)
    model = YOLO(weights)
    demo = build_demo(model)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())