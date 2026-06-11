"""
inference.py — Med-GaMMa inference.

Usage:
    python -m src.inference --image path/to/slide.jpg --question "What is present?"
    python -m src.inference --image-url https://... --question "Is this benign?"
    python -m src.inference --model outputs/final --image slide.jpg --question "..." --temperature 0.3
"""

import os
import sys
import argparse
from PIL import Image
import requests
from io import BytesIO
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = (
    "You are an expert pathologist with decades of experience in diagnostic histopathology. "
    "Analyze the pathology image carefully and answer the clinical question with a precise, "
    "evidence-based explanation grounded in what is visible in the image."
)


def load_model(model_path: str, load_in_4bit: bool = True):
    """
    Load a fine-tuned Med-GaMMa (LoRA/DoRA adapter merged or standalone).
    Accepts both local directory paths and HuggingFace Hub model IDs.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    print(f"[inference] Loading model: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path)

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
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
    print(f"[inference] Model on {next(model.parameters()).device}")
    return model, processor


def load_image(image_path: str = None, image_url: str = None) -> Image.Image:
    """Load a PIL image from a local path or URL."""
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image_url:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    raise ValueError("Provide either image_path or image_url")


def predict(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.1,
    instruction: str = SYSTEM_PROMPT,
) -> str:
    """
    Run single-image pathology VQA inference.

    Args:
        model: loaded Med-GaMMa model
        processor: corresponding AutoProcessor
        image: PIL Image (RGB)
        question: clinical question string
        max_new_tokens: max answer tokens to generate
        temperature: sampling temperature (used when do_sample=True)
        top_p: nucleus sampling p (used when do_sample=True)
        do_sample: False = greedy/beam, True = sampling
        repetition_penalty: penalise repeated tokens (>1.0 reduces repetition)
        instruction: system prompt

    Returns:
        str: model answer text
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

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    answer = processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()
    return answer


def predict_batch(
    model,
    processor,
    images: list,
    questions: list,
    max_new_tokens: int = 256,
    instruction: str = SYSTEM_PROMPT,
) -> list[str]:
    """
    Run batch inference for multiple image+question pairs.
    Useful for evaluation loops — avoids per-sample overhead.
    """
    assert len(images) == len(questions), "images and questions must have equal length"

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ],
            }
        ]
        for img, q in zip(images, questions)
    ]

    answers = []
    for conv, img in zip(conversations, images):
        inputs = processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[1]
        answers.append(
            processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        )

    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Med-GaMMa PathVQA inference")
    parser.add_argument("--model", default="outputs/final")
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-url", default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bf16 instead of 4-bit")
    args = parser.parse_args()

    model, processor = load_model(args.model, load_in_4bit=not args.no_4bit)
    image = load_image(image_path=args.image, image_url=args.image_url)
    answer = predict(
        model, processor, image, args.question,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.sample,
    )
    print(f"\nQuestion : {args.question}")
    print(f"Answer   : {answer}")
