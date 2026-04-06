"""
inference.py — Med-GaMMa inference with standard HuggingFace transformers + PEFT.

Usage:
    python src/inference.py --image path/to/image.jpg --question "What is present?"
    python src/inference.py --image-url https://... --question "Is this benign?"
"""

import os
import sys
import argparse
from PIL import Image
import requests
from io import BytesIO
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model(model_path: str, load_in_4bit: bool = True):
    """Load fine-tuned Med-GaMMa (LoRA adapter) from local path."""
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    print(f"[inference] Loading model from {model_path}...")

    processor = AutoProcessor.from_pretrained(model_path)

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    print(f"[inference] Model loaded on {next(model.parameters()).device}")
    return model, processor


def load_image(image_path: str = None, image_url: str = None) -> Image.Image:
    """Load image from local path or URL."""
    if image_path:
        return Image.open(image_path).convert("RGB")
    elif image_url:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise ValueError("Provide either image_path or image_url")


def predict(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 256,
    instruction: str = "You are an expert pathologist. Analyze the pathology image carefully and answer the clinical question with a detailed, accurate explanation.",
) -> str:
    """
    Run inference on a single image + question.

    Returns:
        str: model's answer
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    answer = processor.tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/final")
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-url", default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    model, processor = load_model(args.model)
    image = load_image(image_path=args.image, image_url=args.image_url)
    answer = predict(model, processor, image, args.question, args.max_tokens)

    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {answer}")
