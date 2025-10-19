from unsloth import FastVisionModel
from src.data_processing import main as load_dataset

def test_model_inference():
    # Load small portion of dataset
    train_dataset, _ = load_dataset()
    sample = train_dataset[0]

    # Load pretrained model (small for test)
    model, processor = FastVisionModel.from_pretrained(
        "unsloth/gemma-3n-E2B-it",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map="auto"
    )
    model.eval()

    # Perform a single forward pass
    conversation = sample["messages"]
    outputs = model.generate(conversation, max_new_tokens=32)
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    assert isinstance(answer, str) and len(answer) > 0, "Model inference failed or returned empty answer"
    print("✅ Model loaded and inference successful.")
    print("Sample answer:", answer)

if __name__ == "__main__":
    test_model_inference()
