from datasets import load_dataset

instruction = "You are an expert pathologist. Answer the question based on the image."

def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]}
                ]
            }
        ]
    }

def main():
    # Load Path-VQA directly from HF
    dataset = load_dataset("moebouassida/enhanced_path-vqa")

    # Convert train and val splits
    train_conv = [convert_to_conversation(s) for s in dataset["train"]]
    val_conv = [convert_to_conversation(s) for s in dataset["val"]]

    print(f"Train samples: {len(train_conv)}, Val samples: {len(val_conv)}")

    return train_conv, val_conv

if __name__ == "__main__":
    train_conv, val_conv = main()
