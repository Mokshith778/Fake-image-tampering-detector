
"""
Professional Gradio interface for AI image tampering detection.

Run:
    python app.py --checkpoint runs/casia_r34/best.pt --port 7861
"""

import argparse
import os

import gradio as gr
import numpy as np

from tamper_detector.ela import compute_ela, ela_for_display, to_rgb
from tamper_detector.inference import TamperDetector


# ==================================================
# 1. CONFIGURATION
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    default=os.environ.get(
        "TAMPER_CKPT",
        "runs/exp/best.pt"
    )
)

parser.add_argument("--share", action="store_true")
parser.add_argument("--port", type=int, default=7860)

args = parser.parse_args()

det = None

if os.path.exists(args.checkpoint):

    det = TamperDetector(args.checkpoint)

    print(f"[app] Model loaded: {args.checkpoint}")
    print(f"[app] Device: {det.device}")

else:

    print(f"[app] Checkpoint not found: {args.checkpoint}")
    print("[app] Running in ELA-only mode")


# ==================================================
# 2. IMAGE ANALYSIS
# ==================================================

def analyse(img, pixel_threshold):

    if img is None:
        return (
            None, None, None, None, None,
            "Please upload an image first."
        )

    try:

        # Convert input into RGB
        img = to_rgb(img)
        original = np.asarray(img).copy()

        # ------------------------------------------
        # ELA-ONLY FALLBACK
        # ------------------------------------------

        if det is None:

            ela = ela_for_display(
                compute_ela(img)
            )

            return (
                original,
                ela,
                None,
                None,
                {"ELA-only mode": 1.0},
                "No trained checkpoint loaded. Only ELA analysis is available."
            )

        # ------------------------------------------
        # MODEL INFERENCE
        # ------------------------------------------

        r = det.predict(img, pixel_threshold)

        verdict = (
            "TAMPERED"
            if r.tampered
            else "AUTHENTIC"
        )

        other = (
            "AUTHENTIC"
            if r.tampered
            else "TAMPERED"
        )

        confidence = r.confidence

        tampered_area = (
            100.0 * float(r.mask.mean())
        )

        label = {
            verdict: confidence,
            other: 1.0 - confidence
        }

        # ------------------------------------------
        # BINARY MASK VISUALIZATION
        # ------------------------------------------

        binary_mask = (
            r.mask.astype(np.uint8) * 255
        )

        # ------------------------------------------
        # FORENSIC REPORT
        # ------------------------------------------

        report = f"""
## Forensic Analysis Report

### Detection Result: {verdict}

| Metric | Value |
|:---|---:|
| Tampering score | **{r.score:.4f}** |
| Verdict confidence | **{confidence:.2%}** |
| Image decision threshold | {r.threshold:.4f} |
| Pixel threshold | {pixel_threshold:.2f} |
| Estimated affected area | **{tampered_area:.2f}%** |
| Processing device | {det.device.upper()} |

### Interpretation

{"The model identified suspicious regions in the uploaded image." if r.tampered else "The model classified the uploaded image as authentic."}

The affected area is calculated from the predicted binary segmentation mask.

**Important:** These results are machine-learning predictions, not definitive proof of image authenticity or manipulation.
"""

        return (
            original,
            r.ela,
            r.overlay,
            binary_mask,
            label,
            report
        )

    except Exception as e:

        print(f"[app] Analysis error: {e}")

        return (
            None,
            None,
            None,
            None,
            None,
            f"### Analysis failed\n\n`{str(e)}`"
        )


# ==================================================
# 3. DASHBOARD CSS
# ==================================================

CSS = """

/* Main application */

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

/* Header */

#main-title {
    text-align: center;
    padding: 24px 0;
}

#main-title h1 {
    font-size: 34px;
    font-weight: 750;
}

/* Section spacing */

.section-heading {
    margin-top: 18px;
    margin-bottom: 12px;
}

/* Upload image component */

#upload-image {
    min-height: 350px;
}

#upload-image img {
    object-fit: contain !important;
}

/* Visualization panels */

#visual-grid .image-container {
    height: 320px !important;
    min-height: 320px !important;
}

#visual-grid img {
    object-fit: contain !important;
}

/* Image panel sizing */

#visual-grid > div {
    min-width: 0 !important;
}

/* Analyze button */

#analyze-button {
    min-height: 50px;
    font-weight: 700;
    font-size: 16px;
}

/* Report */

#analysis-report {
    line-height: 1.7;
}

/* Responsive layout */

@media (max-width: 768px) {

    #main-title h1 {
        font-size: 25px;
    }

    #visual-grid .image-container {
        height: 260px !important;
        min-height: 260px !important;
    }

}

"""


# ==================================================
# 4. GRADIO APPLICATION
# ==================================================

with gr.Blocks(
    title="AI Image Forensics Dashboard"
) as demo:

    # ----------------------------------------------
    # HEADER
    # ----------------------------------------------

    gr.Markdown(
        """
# AI Image Forensics Dashboard

### Deep Learning Based Image Tampering Detection

Detect potential image manipulation using Error Level Analysis
and a U-Net + ResNet34 segmentation model.

---
        """,
        elem_id="main-title"
    )

    # ----------------------------------------------
    # UPLOAD AND RESULTS SECTION
    # ----------------------------------------------

    with gr.Row(equal_height=True):

        # IMAGE UPLOAD

        with gr.Column(scale=1, min_width=300):

            gr.Markdown("## 01 — Upload Image")

            inp = gr.Image(
                type="pil",
                label="Upload image for analysis",
                sources=["upload", "clipboard"],
                interactive=True,
                height=350,
                width="100%",
                container=True,
                elem_id="upload-image"
            )

            thr = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.5,
                step=0.05,
                label="Pixel threshold for localization"
            )

            btn = gr.Button(
                "Analyze Image",
                variant="primary",
                elem_id="analyze-button"
            )

        # DETECTION RESULTS

        with gr.Column(scale=1, min_width=300):

            gr.Markdown("## 02 — Detection Result")

            verdict = gr.Label(
                label="Authenticity Classification",
                num_top_classes=2
            )

            report = gr.Markdown(
                """
Upload an image and click **Analyze Image**
to generate a forensic analysis report.
                """,
                elem_id="analysis-report"
            )

    # ----------------------------------------------
    # FORENSIC VISUALIZATION
    # ----------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        """
## 03 — Forensic Visualization

Compare the original image with the model's
ELA analysis, tampering heatmap and predicted mask.
        """,
        elem_classes=["section-heading"]
    )

    # TWO ROWS, TWO COLUMNS

    with gr.Column(elem_id="visual-grid"):

        with gr.Row(equal_height=True):

            original_out = gr.Image(
                label="Original Image",
                type="numpy",
                height=320,
                width="100%",
                scale=1,
                interactive=False
            )

            ela_out = gr.Image(
                label="Error Level Analysis (ELA)",
                type="numpy",
                height=320,
                width="100%",
                scale=1,
                interactive=False
            )

        with gr.Row(equal_height=True):

            heat_out = gr.Image(
                label="Tampering Heatmap",
                type="numpy",
                height=320,
                width="100%",
                scale=1,
                interactive=False
            )

            mask_out = gr.Image(
                label="Predicted Binary Mask",
                type="numpy",
                height=320,
                width="100%",
                scale=1,
                interactive=False
            )

    # ----------------------------------------------
    # ABOUT THE MODEL
    # ----------------------------------------------

    gr.Markdown(
        """
---

## 04 — About the Model

| Component | Configuration |
|:---|:---|
| Architecture | U-Net + ResNet34 |
| Input features | RGB + ELA |
| Dataset | CASIA v2.0 |
| Deep learning framework | PyTorch |
| Hardware acceleration | CUDA (when available) |
| Task | Image classification and pixel-level localization |

### Visualization Guide

- **Original Image:** The image uploaded for forensic examination.
- **ELA:** Highlights differences in JPEG compression patterns.
- **Tampering Heatmap:** Shows pixel-level suspiciousness predicted by the model.
- **Binary Mask:** Shows pixels exceeding the selected localization threshold.

### Disclaimer

Detection results are estimates produced by a machine learning model.
They should not be treated as conclusive forensic evidence.
        """
    )


    # ----------------------------------------------
    # CONNECT ANALYSIS FUNCTION
    # ----------------------------------------------

    btn.click(
        fn=analyse,
        inputs=[
            inp,
            thr
        ],
        outputs=[
            original_out,
            ela_out,
            heat_out,
            mask_out,
            verdict,
            report
        ]
    )


# ==================================================
# 5. LAUNCH APPLICATION
# ==================================================

if __name__ == "__main__":

    demo.launch(
        share=args.share,
        server_port=args.port,
        css=CSS,
        theme=gr.themes.Soft()
    )
