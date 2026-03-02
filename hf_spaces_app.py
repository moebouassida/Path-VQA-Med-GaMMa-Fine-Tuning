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

model     = None
processor = None

if MODEL_PATH.exists():
    from src.inference import load_model
    model, processor = load_model(str(MODEL_PATH))
    print(f"[HF Space] Model loaded from {MODEL_PATH}")
else:
    print(f"[HF Space] No model at {MODEL_PATH}")


def answer_question(image, question: str, max_tokens: int = 256) -> str:
    if image is None:
        return "Please upload a pathology image."
    if not question or question.strip() == "":
        return "Please enter a clinical question."
    if model is None:
        return "Model not loaded. Please check deployment."
    try:
        from src.inference import predict
        return predict(model, processor, image, question, max_tokens)
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="Path-VQA Med-GaMMa") as demo:
    gr.Markdown("""
    # Pathology Visual Question Answering
    **Med-GaMMa fine-tuned on PathVQA Enhanced** — upload a pathology image and ask a clinical question.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input    = gr.Image(type="pil", label="Pathology Image")
            question_input = gr.Textbox(label="Clinical Question",
                                        placeholder="e.g. What type of tissue is shown?")
            max_tokens     = gr.Slider(64, 512, value=256, step=32, label="Max Response Length")
            run_btn        = gr.Button("Ask Question", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)

    run_btn.click(fn=answer_question,
                  inputs=[image_input, question_input, max_tokens],
                  outputs=[answer_output])

    gr.Markdown("---\n**Model:** Med-GaMMa 4B (LoRA) · **Dataset:** PathVQA Enhanced · **Author:** Moez Bouassida")

if __name__ == "__main__":
    demo.launch()