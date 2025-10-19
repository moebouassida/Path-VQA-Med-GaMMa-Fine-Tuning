from data_preprocessing import main as load_dataset

def test_dataset_loading():
    train_dataset, val_dataset = load_dataset()

    assert len(train_dataset) > 0, "Train dataset is empty!"
    assert len(val_dataset) > 0, "Validation dataset is empty!"

    sample = train_dataset[0]
    # Check required keys
    assert "messages" in sample, "'messages' key missing in dataset sample"
    assert isinstance(sample["messages"], list), "'messages' should be a list"
    assert all("role" in msg and "content" in msg for msg in sample["messages"]), "Each message must have 'role' and 'content'"

    print("✅ Dataset loaded correctly with expected format.")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

if __name__ == "__main__":
    test_dataset_loading()
