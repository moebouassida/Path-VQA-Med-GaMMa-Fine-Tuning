"""
HuggingFace Spaces — PathVQA · Med-GaMMa

Production Gradio demo for pathology Visual Question Answering.
Med-GaMMa 4B fine-tuned on PathVQA Enhanced (~32K QA pairs).
"""

import os
import sys
import time
from pathlib import Path

import gradio as gr
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Model config ───────────────────────────────────────────────────────────────
HF_MODEL_ID = os.getenv("MODEL_ID", "moebouassida/medgemma-4b-path-vqa")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/final"))

_model = None
_processor = None
_load_error: str = ""


def _try_load():
    global _model, _processor, _load_error
    from src.inference import load_model
    try:
        src = str(MODEL_PATH) if MODEL_PATH.exists() else HF_MODEL_ID
        _model, _processor = load_model(src, load_in_4bit=True)
        print(f"[demo] Model loaded from: {src}")
    except Exception as e:
        _load_error = str(e)
        print(f"[demo] Could not load model: {e}")


_try_load()

# ── Pre-load dataset examples ─────────────────────────────────────────────────
_examples: list = []
try:
    from datasets import load_dataset as _hf_load
    _ex_ds = _hf_load("moebouassida/path-vqa-enhanced", split="test[:8]")
    _examples = [[s["image"], s["question"]] for s in _ex_ds]
    print(f"[demo] Loaded {len(_examples)} examples from dataset")
except Exception as e:
    print(f"[demo] Could not pre-load examples: {e}")

# ── Quick question shortcuts ──────────────────────────────────────────────────
QUICK_QUESTIONS = [
    "What type of tissue is shown?",
    "Is there evidence of malignancy?",
    "What abnormalities are visible?",
    "Is this tissue benign or malignant?",
    "Describe the cellular architecture.",
    "What is the mitotic activity?",
    "Are inflammatory cells present?",
    "What is the likely diagnosis?",
]


# ── Inference function ─────────────────────────────────────────────────────────

def run_vqa(image: Image.Image, question: str, max_tokens: int = 256) -> tuple[str, str]:
    """
    Returns (answer_text, answer_type_html).
    answer_type_html is a small HTML badge shown above the answer.
    """
    if image is None:
        return "Please upload a pathology image.", ""
    if not question or not question.strip():
        return "Please enter a clinical question.", ""
    if _model is None:
        msg = _load_error or "Model not loaded — check space logs."
        return f"Model unavailable: {msg}", ""

    try:
        from src.inference import predict
        t0 = time.time()
        answer = predict(_model, _processor, image, question.strip(), max_tokens)
        elapsed = time.time() - t0

        norm = answer.strip().lower()
        if norm in ("yes", "no") or norm.startswith(("yes,", "no,", "yes.", "no.")):
            badge_color = "#16a34a"
            badge_label = "Yes / No"
        else:
            badge_color = "#2563eb"
            badge_label = "Open-ended"

        type_html = (
            f'<div style="margin-bottom:8px">'
            f'<span style="background:{badge_color};color:white;padding:3px 12px;'
            f'border-radius:20px;font-size:0.82em;font-weight:600">{badge_label}</span>'
            f'&nbsp;&nbsp;<span style="color:#6b7280;font-size:0.8em">⏱ {elapsed:.1f}s</span>'
            f'</div>'
        )
        return answer, type_html

    except Exception as e:
        return f"Inference error: {e}", ""


# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
/* Hide Gradio footer */
footer { display: none !important; }

/* Answer text box */
.answer-box textarea {
    font-size: 1.05em;
    line-height: 1.7;
    background: #f0f9ff;
    border: 1px solid #bae6fd;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 18px;
}

/* Stats cards */
.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
"""

# ── Hero HTML ──────────────────────────────────────────────────────────────────
_HERO = """
<div class="hero-banner">
  <h1 style="color:white;margin:0 0 6px 0;font-size:1.9em;font-weight:700">
    🔬 PathVQA · Med-GaMMa
  </h1>
  <p style="color:#93c5fd;margin:0;font-size:1.05em">
    Visual Question Answering on Pathology Images —
    <strong style="color:#dbeafe">Med-GaMMa 4B</strong>
    fine-tuned on <a href="https://huggingface.co/datasets/moebouassida/path-vqa-enhanced"
    style="color:#60a5fa">PathVQA Enhanced</a> (~32K QA pairs)
  </p>
</div>
"""

# ── Stats sidebar ──────────────────────────────────────────────────────────────
_STATS = """
### Model Card

| | |
|:--|:--|
| **Base model** | Med-GaMMa 4B (`google/medgemma-4b-it`) |
| **Adapter** | DoRA + RSLoRA (r=16) |
| **Attention** | Flash Attention 2 |
| **Quantization** | 4-bit NF4, double quant |
| **Dataset** | PathVQA Enhanced (~32K pairs) |
| **GPU** | NVIDIA RTX 4090 · 24 GB |

### Benchmark

| Metric | Score |
|:--|--:|
| Yes/No Accuracy | **~76%** |
| Open-ended BLEU-4 | **~0.28** |
| Open-ended Token F1 | **~0.48** |

<small>Evaluated on PathVQA test split (4,998 samples).</small>
"""

_DISCLAIMER = """
> ⚠️ **Research use only.**
> This tool is not a medical device. It is intended for research and educational purposes only.
> Always consult a qualified pathologist for diagnostic decisions.
"""

_FOOTER = """
<div style="text-align:center;color:#64748b;font-size:0.88em;margin-top:12px">
  <a href="https://huggingface.co/moebouassida/medgemma-4b-path-vqa" style="color:#2563eb">Model</a> ·
  <a href="https://huggingface.co/datasets/moebouassida/path-vqa-enhanced" style="color:#2563eb">Dataset</a> ·
  <a href="https://github.com/moebouassida" style="color:#2563eb">GitHub</a> ·
  Built by <strong>Moez Bouassida</strong>
</div>
"""

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="PathVQA · Med-GaMMa",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=_CSS,
) as demo:

    gr.HTML(_HERO)

    with gr.Row(equal_height=False):

        # ── Left: Input panel ─────────────────────────────────────────────────
        with gr.Column(scale=5, min_width=340):
            image_in = gr.Image(
                type="pil",
                label="Pathology Image (H&E stain, IHC, cytology…)",
                height=300,
            )

            question_in = gr.Textbox(
                label="Clinical Question",
                placeholder=(
                    "e.g.  What type of tissue is shown?\n"
                    "      Is there evidence of malignancy?"
                ),
                lines=2,
            )

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear", variant="secondary", scale=1)
                submit_btn = gr.Button("Analyze ▶", variant="primary", scale=3)

            with gr.Accordion("⚙️ Generation settings", open=False):
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=512, value=256, step=32,
                    label="Max response length (tokens)",
                )

            gr.Markdown("**Quick questions:**")
            with gr.Row(wrap=True):
                for q in QUICK_QUESTIONS[:4]:
                    gr.Button(q, size="sm", scale=0).click(
                        fn=lambda x=q: x,
                        outputs=question_in,
                    )
            with gr.Row(wrap=True):
                for q in QUICK_QUESTIONS[4:]:
                    gr.Button(q, size="sm", scale=0).click(
                        fn=lambda x=q: x,
                        outputs=question_in,
                    )

        # ── Right: Output panel ───────────────────────────────────────────────
        with gr.Column(scale=5, min_width=340):
            answer_type_html = gr.HTML("")
            answer_out = gr.Textbox(
                label="Pathologist AI Answer",
                lines=11,
                interactive=False,
                elem_classes=["answer-box"],
                show_copy_button=True,
            )
            gr.Markdown(_STATS)

    # ── Dataset examples ──────────────────────────────────────────────────────
    if _examples:
        gr.Examples(
            examples=_examples,
            inputs=[image_in, question_in],
            label="📂 Examples from PathVQA test split",
            examples_per_page=4,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(_DISCLAIMER)
    gr.HTML(_FOOTER)

    # ── Event wiring ──────────────────────────────────────────────────────────
    submit_btn.click(
        fn=run_vqa,
        inputs=[image_in, question_in, max_tokens_slider],
        outputs=[answer_out, answer_type_html],
    )
    question_in.submit(
        fn=run_vqa,
        inputs=[image_in, question_in, max_tokens_slider],
        outputs=[answer_out, answer_type_html],
    )
    clear_btn.click(
        fn=lambda: (None, "", "", ""),
        outputs=[image_in, question_in, answer_out, answer_type_html],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
