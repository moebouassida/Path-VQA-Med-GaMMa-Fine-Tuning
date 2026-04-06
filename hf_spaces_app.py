"""
HuggingFace Spaces entry point — Gradio demo for Path-VQA Med-GaMMa.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/final"))

model = None
processor = None

if MODEL_PATH.exists():
    from src.inference import load_model

    model, processor = load_model(str(MODEL_PATH))
    print(f"[demo] Model loaded from {MODEL_PATH}")
else:
    print(f"[demo] No model at {MODEL_PATH} — running in demo mode")

EXAMPLE_QUESTIONS = [
    "What type of tissue is shown in this image?",
    "Is there evidence of malignancy?",
    "What abnormalities are visible?",
    "Is this tissue benign or malignant?",
    "What is the mitotic activity in this sample?",
]


def answer_question(image, question: str, max_tokens: int = 256):
    if image is None:
        return "⚠️ Please upload a pathology image.", ""
    if not question or question.strip() == "":
        return "⚠️ Please enter a clinical question.", ""
    if model is None:
        return (
            "⚠️ Model not loaded. Train the model first or set MODEL_PATH to a valid checkpoint.",
            "",
        )
    try:
        from src.inference import predict

        answer = predict(model, processor, image, question, max_tokens)

        # Detect yes/no vs open-ended
        normalized = answer.strip().lower()
        if normalized in ("yes", "no") or normalized.startswith(("yes,", "no,")):
            answer_type = "Yes/No"
        else:
            answer_type = "Open-ended"

        return answer, f"Answer type: **{answer_type}**"
    except Exception as e:
        return f"Error: {str(e)}", ""


with gr.Blocks(
    title="Path-VQA · Med-GaMMa",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# 🔬 Pathology Visual Question Answering
**Med-GaMMa 4B** fine-tuned on [PathVQA Enhanced](https://huggingface.co/datasets/moebouassida/path-vqa-enhanced) —
upload a pathology image and ask a clinical question.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Pathology Image")

            question_input = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g. What type of tissue is shown?",
                lines=2,
            )

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                run_btn = gr.Button("Ask Question ▶", variant="primary", scale=2)

            max_tokens = gr.Slider(
                64, 512, value=256, step=32, label="Max Response Length"
            )

            gr.Markdown("**Example questions:**")
            for q in EXAMPLE_QUESTIONS:
                gr.Button(q, size="sm").click(
                    fn=lambda x=q: x,
                    outputs=question_input,
                )

        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Answer", lines=10, interactive=False
            )
            answer_type_output = gr.Markdown("")

    run_btn.click(
        fn=answer_question,
        inputs=[image_input, question_input, max_tokens],
        outputs=[answer_output, answer_type_output],
    )

    clear_btn.click(
        fn=lambda: (None, "", "", ""),
        outputs=[image_input, question_input, answer_output, answer_type_output],
    )

    gr.Markdown(
        """
---
| | |
|---|---|
| **Model** | Med-GaMMa 4B · LoRA (r=32) |
| **Dataset** | PathVQA Enhanced (moebouassida/path-vqa-enhanced) |
| **Tasks** | Yes/No · Open-ended pathology VQA |
| **Author** | [Moez Bouassida](https://github.com/moebouassida) |

> ⚠️ Research use only. Not a medical device.
"""
    )

if __name__ == "__main__":
    demo.launch()
