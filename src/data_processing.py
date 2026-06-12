from datasets import load_dataset

INSTRUCTION = (
    "You are an expert pathologist. "
    "Analyze the pathology image carefully and answer the clinical question "
    "with a detailed, accurate explanation."
)


def convert_to_conversation(sample: dict, use_enhanced: bool = True) -> dict:
    answer_text = sample["enhanced_answer"] if use_enhanced else sample["answer"]
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
    print(f"[data] Loading {dataset_name}...")
    dataset = load_dataset(dataset_name)

    train_split = dataset["train"].shuffle(seed=42)
    val_split = dataset["validation"]

    if max_train_samples:
        train_split = train_split.select(
            range(min(max_train_samples, len(train_split)))
        )
    if max_val_samples:
        val_split = val_split.select(range(min(max_val_samples, len(val_split))))

    yes_no = sum(1 for s in train_split if str(s["answer"]).strip().lower() in ("yes", "no"))
    print(
        f"[data] Yes/No questions: {yes_no}/{len(train_split)} "
        f"({100 * yes_no / max(len(train_split), 1):.1f}%)"
    )

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
