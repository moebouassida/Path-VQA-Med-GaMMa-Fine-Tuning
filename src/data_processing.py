"""
data_processing.py — Load and format PathVQA Enhanced dataset for Med-GaMMa fine-tuning.

Dataset: moebouassida/path-vqa-enhanced
    - image: PIL Image
    - question: str
    - answer: str (short original answer)
    - enhanced_answer: str (clinically detailed explanation)

Conversation format for Med-GaMMa (Gemma 3 chat template):
    user:      [instruction] + [image] + [question]
    assistant: [enhanced_answer]
"""

from datasets import load_dataset

INSTRUCTION = (
    "You are an expert pathologist. "
    "Analyze the pathology image carefully and answer the clinical question "
    "with a detailed, accurate explanation."
)


def convert_to_conversation(sample: dict, use_enhanced: bool = True) -> dict:
    """
    Convert a PathVQA sample to Med-GaMMa conversation format.

    Args:
        sample: dict with keys: image, question, answer, enhanced_answer
        use_enhanced: if True, use enhanced_answer (richer); else use answer
    """
    answer_text = sample["enhanced_answer"] if use_enhanced else sample["answer"]

    # Fallback to short answer if enhanced is empty
    if not answer_text or str(answer_text).strip() == "":
        answer_text = sample["answer"]

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(answer_text)},
                ],
            },
        ]
    }


def main(
    dataset_name: str = "moebouassida/path-vqa-enhanced",
    use_enhanced: bool = True,
    max_train_samples: int = None,
    max_val_samples: int = None,
):
    """
    Load PathVQA Enhanced and return train/val conversation lists.

    Returns:
        (train_conversations, val_conversations)
    """
    print(f"[data] Loading {dataset_name}...")
    dataset = load_dataset(dataset_name)

    train_split = dataset["train"]
    val_split = dataset["validation"]

    if max_train_samples:
        train_split = train_split.select(
            range(min(max_train_samples, len(train_split)))
        )
    if max_val_samples:
        val_split = val_split.select(range(min(max_val_samples, len(val_split))))

    print(f"[data] Converting {len(train_split)} train samples...")
    train_conv = [convert_to_conversation(s, use_enhanced) for s in train_split]

    print(f"[data] Converting {len(val_split)} val samples...")
    val_conv = [convert_to_conversation(s, use_enhanced) for s in val_split]

    print(f"[data] Done — train: {len(train_conv)} | val: {len(val_conv)}")
    return train_conv, val_conv


if __name__ == "__main__":
    train, val = main()
    print("\nSample conversation:")
    print(f"  Question:  {train[0]['messages'][0]['content'][2]['text']}")
    print(f"  Answer:    {train[0]['messages'][1]['content'][0]['text'][:100]}...")
