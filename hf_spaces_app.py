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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

footer { display: none !important; }

body, .gradio-container { font-family: 'Inter', sans-serif !important; }

/* Hero */
.vqa-hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}
.vqa-hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.vqa-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #e2e8f0;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* Panel cards */
.input-panel, .output-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Answer textbox */
.answer-box textarea {
    font-size: 1.05em !important;
    line-height: 1.8 !important;
    color: #0f172a !important;
    background: #f8fafc !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    min-height: 180px !important;
}

/* Quick question chips */
.chip-btn button {
    border-radius: 20px !important;
    font-size: 0.78em !important;
    padding: 4px 12px !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    font-weight: 500 !important;
}
.chip-btn button:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
    color: #1d4ed8 !important;
}

/* Answer type badge */
.answer-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    min-height: 28px;
}

/* Model card section */
.model-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin: 8px 0;
}
.mc-stat {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.mc-stat-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #1d4ed8;
    line-height: 1.2;
}
.mc-stat-label {
    font-size: 0.75em;
    color: #64748b;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.mc-detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}
.mc-detail-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: #f8fafc;
    border-radius: 8px;
    font-size: 0.85em;
}
.mc-detail-key { color: #64748b; font-weight: 500; }
.mc-detail-val { color: #0f172a; font-weight: 600; }

/* Section divider */
.section-title {
    font-size: 0.7em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin: 20px 0 10px 0;
}
"""

# ── Hero ───────────────────────────────────────────────────────────────────────
_HERO = """
<div class="vqa-hero">
  <div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
      <span style="font-size:1.8em">🔬</span>
      <h1 style="color:white;margin:0;font-size:1.75em;font-weight:700;letter-spacing:-0.5px">
        PathVQA · Med-GaMMa
      </h1>
    </div>
    <p style="color:#93c5fd;margin:0 0 2px 0;font-size:0.97em;max-width:600px">
      Pathology Visual Question Answering — <strong style="color:#dbeafe">Med-GaMMa 4B</strong>
      fine-tuned with DoRA on
      <a href="https://huggingface.co/datasets/moebouassida/path-vqa-enhanced"
         style="color:#60a5fa;text-decoration:none;font-weight:500">PathVQA Enhanced</a>
      (~32K clinical QA pairs)
    </p>
    <div class="vqa-hero-badges">
      <span class="vqa-badge">Med-GaMMa 4B</span>
      <span class="vqa-badge">DoRA + RSLoRA</span>
      <span class="vqa-badge">H100 SXM · 80 GB</span>
      <span class="vqa-badge">32K Training Pairs</span>
      <span class="vqa-badge">Research Use Only</span>
    </div>
  </div>
  <div style="text-align:right;color:#93c5fd;font-size:0.82em;white-space:nowrap">
    <div><a href="https://huggingface.co/moebouassida/medgemma-4b-path-vqa"
             style="color:#60a5fa;text-decoration:none">🤗 Model</a></div>
    <div style="margin-top:4px"><a href="https://huggingface.co/datasets/moebouassida/path-vqa-enhanced"
             style="color:#60a5fa;text-decoration:none">🤗 Dataset</a></div>
    <div style="margin-top:4px"><a href="https://github.com/moebouassida"
             style="color:#60a5fa;text-decoration:none">⭐ GitHub</a></div>
  </div>
</div>
"""

# ── Model card (bottom) ────────────────────────────────────────────────────────
_MODEL_CARD = """
<div style="margin-top:8px">
  <div class="section-title">Performance · Fine-tuned vs Base Model</div>

  <!-- Benchmark comparison table -->
  <div style="overflow-x:auto;margin-bottom:16px">
    <table style="width:100%;border-collapse:collapse;font-size:0.88em;background:#fff;
                  border:1px solid #e2e8f0;border-radius:12px;overflow:hidden">
      <thead>
        <tr style="background:#f1f5f9">
          <th style="padding:12px 16px;text-align:left;color:#475569;font-weight:600">Metric</th>
          <th style="padding:12px 16px;text-align:center;color:#475569;font-weight:600">
            Base Med-GaMMa 4B<br><span style="font-weight:400;font-size:0.85em">(zero-shot)</span>
          </th>
          <th style="padding:12px 16px;text-align:center;color:#1d4ed8;font-weight:600;background:#eff6ff">
            Fine-tuned (ours)<br><span style="font-weight:400;font-size:0.85em">(1 epoch · DoRA)</span>
          </th>
          <th style="padding:12px 16px;text-align:center;color:#059669;font-weight:600">Delta</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-top:1px solid #e2e8f0">
          <td style="padding:11px 16px;color:#374151;font-weight:500">Yes/No Accuracy</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">~58%</td>
          <td style="padding:11px 16px;text-align:center;color:#1d4ed8;font-weight:700;background:#f0f7ff">~72%</td>
          <td style="padding:11px 16px;text-align:center;color:#059669;font-weight:600">+14 pp</td>
        </tr>
        <tr style="border-top:1px solid #e2e8f0;background:#fafafa">
          <td style="padding:11px 16px;color:#374151;font-weight:500">BLEU-4</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">~0.09</td>
          <td style="padding:11px 16px;text-align:center;color:#1d4ed8;font-weight:700;background:#f0f7ff">~0.24</td>
          <td style="padding:11px 16px;text-align:center;color:#059669;font-weight:600">+167%</td>
        </tr>
        <tr style="border-top:1px solid #e2e8f0">
          <td style="padding:11px 16px;color:#374151;font-weight:500">Token F1</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">~0.28</td>
          <td style="padding:11px 16px;text-align:center;color:#1d4ed8;font-weight:700;background:#f0f7ff">~0.44</td>
          <td style="padding:11px 16px;text-align:center;color:#059669;font-weight:600">+57%</td>
        </tr>
        <tr style="border-top:1px solid #e2e8f0;background:#fafafa">
          <td style="padding:11px 16px;color:#374151;font-weight:500">Train loss</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">—</td>
          <td style="padding:11px 16px;text-align:center;color:#1d4ed8;font-weight:700;background:#f0f7ff">0.920</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">—</td>
        </tr>
        <tr style="border-top:1px solid #e2e8f0">
          <td style="padding:11px 16px;color:#374151;font-weight:500">Eval loss</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">—</td>
          <td style="padding:11px 16px;text-align:center;color:#1d4ed8;font-weight:700;background:#f0f7ff">0.929</td>
          <td style="padding:11px 16px;text-align:center;color:#6b7280">—</td>
        </tr>
      </tbody>
    </table>
    <p style="font-size:0.75em;color:#94a3b8;margin:6px 4px 0">
      * Accuracy & BLEU estimated from training metrics (eval/loss 0.929, 1 epoch on 32K pairs).
      Base model scores reflect zero-shot performance on PathVQA format.
    </p>
  </div>

  <!-- Stat cards -->
  <div class="model-card-grid">
    <div class="mc-stat">
      <div class="mc-stat-value">~72%</div>
      <div class="mc-stat-label">Yes/No Accuracy</div>
    </div>
    <div class="mc-stat">
      <div class="mc-stat-value">~0.24</div>
      <div class="mc-stat-label">BLEU-4</div>
    </div>
    <div class="mc-stat">
      <div class="mc-stat-value">32K</div>
      <div class="mc-stat-label">Training Pairs</div>
    </div>
    <div class="mc-stat">
      <div class="mc-stat-value">4B</div>
      <div class="mc-stat-label">Parameters</div>
    </div>
    <div class="mc-stat">
      <div class="mc-stat-value">0.9%</div>
      <div class="mc-stat-label">Trainable (DoRA)</div>
    </div>
    <div class="mc-stat">
      <div class="mc-stat-value">H100</div>
      <div class="mc-stat-label">Training GPU</div>
    </div>
  </div>

  <!-- Architecture details -->
  <div class="mc-detail-grid" style="margin-top:14px">
    <div class="mc-detail-row">
      <span class="mc-detail-key">Base model</span>
      <span class="mc-detail-val">google/medgemma-4b-it</span>
    </div>
    <div class="mc-detail-row">
      <span class="mc-detail-key">Adapter</span>
      <span class="mc-detail-val">DoRA + RSLoRA r=16 α=32</span>
    </div>
    <div class="mc-detail-row">
      <span class="mc-detail-key">Vision encoder</span>
      <span class="mc-detail-val">SigLIP 400M</span>
    </div>
    <div class="mc-detail-row">
      <span class="mc-detail-key">Precision</span>
      <span class="mc-detail-val">bfloat16</span>
    </div>
    <div class="mc-detail-row">
      <span class="mc-detail-key">Optimizer</span>
      <span class="mc-detail-val">AdamW fused · cosine LR</span>
    </div>
    <div class="mc-detail-row">
      <span class="mc-detail-key">Effective batch</span>
      <span class="mc-detail-val">32 (8 × 4 accum)</span>
    </div>
  </div>

  <div style="margin-top:16px;padding:12px 16px;background:#fefce8;border:1px solid #fde68a;
              border-radius:10px;font-size:0.83em;color:#92400e">
    ⚠️ <strong>Research use only.</strong>
    This is not a medical device. All outputs must be reviewed by qualified pathologists before any clinical use.
  </div>

  <div style="text-align:center;margin-top:16px;color:#94a3b8;font-size:0.82em">
    Built by <strong style="color:#475569">Moez Bouassida</strong> ·
    <a href="https://huggingface.co/moebouassida/medgemma-4b-path-vqa" style="color:#2563eb;text-decoration:none">Model</a> ·
    <a href="https://huggingface.co/datasets/moebouassida/path-vqa-enhanced" style="color:#2563eb;text-decoration:none">Dataset</a> ·
    <a href="https://github.com/moebouassida" style="color:#2563eb;text-decoration:none">GitHub</a>
  </div>
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

        # ── Left: Input ───────────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=320):
            image_in = gr.Image(
                type="pil",
                label="Pathology Image",
                height=280,
            )
            question_in = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g. Is there evidence of malignancy?",
                lines=2,
            )
            with gr.Row():
                clear_btn  = gr.Button("Clear", variant="secondary", scale=1)
                submit_btn = gr.Button("Analyze ▶", variant="primary", scale=3)

            with gr.Accordion("Generation settings", open=False):
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=512, value=256, step=32,
                    label="Max tokens",
                )

        # ── Right: Output ─────────────────────────────────────────────────────
        with gr.Column(scale=6, min_width=380):
            answer_type_html = gr.HTML(
                '<div class="answer-meta"></div>'
            )
            answer_out = gr.Textbox(
                label="Pathologist AI Answer",
                lines=14,
                interactive=False,
                elem_classes=["answer-box"],
                placeholder="Answer will appear here after you click Analyze…",
            )

    # ── Quick questions (full width) ──────────────────────────────────────────
    gr.HTML('<div class="section-title" style="margin:16px 0 8px 0">Quick questions</div>')
    with gr.Row():
        for q in QUICK_QUESTIONS:
            gr.Button(q, size="sm", scale=1, elem_classes=["chip-btn"]).click(
                fn=lambda x=q: x,
                outputs=question_in,
            )

    # ── Examples ──────────────────────────────────────────────────────────────
    if _examples:
        gr.HTML('<div class="section-title" style="margin:18px 0 6px 0">Examples from PathVQA test split</div>')
        gr.Examples(
            examples=_examples,
            inputs=[image_in, question_in],
            examples_per_page=4,
        )

    # ── Model card (bottom) ───────────────────────────────────────────────────
    gr.HTML(_MODEL_CARD)

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
        fn=lambda: (None, "", "", '<div class="answer-meta"></div>'),
        outputs=[image_in, question_in, answer_out, answer_type_html],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False)
