"""
train.py — Med-GaMMa fine-tuning on PathVQA Enhanced.
Uses HuggingFace native: transformers + peft + trl (no unsloth).

Usage:
    python src/train.py
    python src/train.py --config config/config.yaml
    python src/train.py --config config/config.yaml --smoke-test
    python src/train.py --config config/config.yaml --epochs 3
"""
import os
import sys
import argparse
import yaml
import torch
import wandb
import mlflow
import mlflow.pytorch
from huggingface_hub import login

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import main as load_dataset


def get_collate_fn(processor, max_seq_length):
    """Custom collate function for Med-GaMMa vision+text batches."""

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(
                e["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for e in examples
        ]

        images = []
        for e in examples:
            img = None
            for msg in e["messages"]:
                for content in msg["content"]:
                    if content["type"] == "image":
                        img = content["image"]
                        break
                if img:
                    break
            images.append(img)

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def train(cfg: dict, smoke_test: bool = False):
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    print("\n" + "=" * 60)
    print("  Path-VQA Med-GaMMa Fine-Tuning")
    print("=" * 60)
    print("  Model   : google/medgemma-4b-it")
    print(f"  Dataset : {cfg['dataset_name']}")
    print(f"  Epochs  : {cfg['num_train_epochs']}")
    print(f"  LR      : {cfg['learning_rate']}")
    print(f"  LoRA r  : {cfg['lora_r']}")
    print(
        f"  Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print("=" * 60 + "\n")

    # ── HF Login ──────────────────────────────────────────────────
    hf_token = cfg.get("hf_token") or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[auth] HuggingFace login successful")
    else:
        raise ValueError("HF_TOKEN not found in config or environment")

    # ── Data ──────────────────────────────────────────────────────
    max_train = 50 if smoke_test else None
    max_val = 20 if smoke_test else None

    train_dataset, val_dataset = load_dataset(
        dataset_name=cfg["dataset_name"],
        use_enhanced=cfg.get("use_enhanced_answer", True),
        max_train_samples=max_train,
        max_val_samples=max_val,
    )

    # ── Model ─────────────────────────────────────────────────────
    print("[model] Loading Med-GaMMa in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-4b-it",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")

    # ── LoRA ──────────────────────────────────────────────────────
    print("[model] Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # ── Collate fn ────────────────────────────────────────────────
    collate_fn = get_collate_fn(processor, cfg["max_seq_length"])

    # ── W&B ───────────────────────────────────────────────────────
    wandb.init(
        project=cfg.get("wandb_project", "path-vqa-medgemma"),
        entity=cfg.get("wandb_entity"),
        name=f"medgemma_lora_r{cfg['lora_r']}_lr{cfg['learning_rate']}",
        config=cfg,
    )

    # ── MLflow ────────────────────────────────────────────────────
    if cfg.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg.get("mlflow_experiment", "path-vqa-medgemma"))
    mlflow.start_run(run_name=f"lora_r{cfg['lora_r']}")
    mlflow.log_params({k: str(v) for k, v in cfg.items()})

    # ── Trainer ───────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            num_train_epochs=1 if smoke_test else cfg["num_train_epochs"],
            max_steps=5 if smoke_test else -1,
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            max_grad_norm=cfg["max_grad_norm"],
            warmup_steps=cfg["warmup_steps"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            optim=cfg["optim"],
            logging_steps=cfg["logging_steps"],
            save_strategy=cfg["save_strategy"],
            save_steps=cfg.get("save_steps", 100),
            eval_strategy="steps",
            eval_steps=cfg.get("save_steps", 100),
            load_best_model_at_end=True,
            seed=cfg["seed"],
            report_to="wandb",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=cfg["max_seq_length"],
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    try:
        print("[train] Starting training...")
        trainer.train()

        print(f"[train] Saving model to {cfg['output_dir']}/final...")
        trainer.save_model(f"{cfg['output_dir']}/final")
        processor.save_pretrained(f"{cfg['output_dir']}/final")

        mlflow.log_metric("train_loss", trainer.state.log_history[-1].get("loss", 0))
        mlflow.log_artifact(cfg["output_dir"], artifact_path="model")

        print(f"\n{'='*60}")
        print("  Training complete!")
        print(f"  Model saved -> {cfg['output_dir']}/final")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\nInterrupted — saving current state...")
        trainer.save_model(f"{cfg['output_dir']}/interrupted")

    finally:
        mlflow.end_run()
        wandb.finish()

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 5 steps on tiny data to verify pipeline",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["num_train_epochs"] = args.epochs
    if args.lr:
        cfg["learning_rate"] = args.lr

    train(cfg, smoke_test=args.smoke_test)