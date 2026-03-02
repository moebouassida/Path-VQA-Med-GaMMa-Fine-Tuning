"""
train.py — Med-GaMMa fine-tuning on PathVQA Enhanced with W&B + MLflow.

Usage:
    python src/train.py
    python src/train.py --config config/config.yaml
    python src/train.py --config config/config.yaml --epochs 3 --smoke-test
"""
import os
import sys
import argparse
import yaml
import wandb
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import main as load_dataset


def train(cfg: dict, smoke_test: bool = False):
    # ── Lazy imports (heavy ML libs) ──────────────────────────────
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    print("\n" + "="*60)
    print("  Path-VQA Med-GaMMa Fine-Tuning")
    print("="*60)
    print(f"  Model   : {cfg['pretrained_model']}")
    print(f"  Dataset : {cfg['dataset_name']}")
    print(f"  Epochs  : {cfg['num_train_epochs']}")
    print(f"  LR      : {cfg['learning_rate']}")
    print(f"  LoRA r  : {cfg['lora_r']}")
    print("="*60 + "\n")

    # ── Data ──────────────────────────────────────────────────────
    max_train = 50 if smoke_test else None
    max_val   = 20 if smoke_test else None

    train_dataset, val_dataset = load_dataset(
        dataset_name=cfg["dataset_name"],
        use_enhanced=cfg.get("use_enhanced_answer", True),
        max_train_samples=max_train,
        max_val_samples=max_val,
    )

    # ── Model ─────────────────────────────────────────────────────
    print("[model] Loading pretrained Med-GaMMa...")
    model, processor = FastVisionModel.from_pretrained(
        cfg["pretrained_model"],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map="auto",
    )

    print("[model] Applying LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        random_state=cfg["seed"],
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # ── Data Collator ─────────────────────────────────────────────
    data_collator = UnslothVisionDataCollator(
        model, processor, resize=cfg["resize"]
    )

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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
        data_collator=data_collator,
        args=SFTConfig(
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=cfg["max_grad_norm"],
            warmup_steps=cfg["warmup_steps"],
            num_train_epochs=1 if smoke_test else cfg["num_train_epochs"],
            max_steps=5 if smoke_test else -1,
            learning_rate=cfg["learning_rate"],
            logging_steps=cfg["logging_steps"],
            save_strategy=cfg["save_strategy"],
            save_steps=cfg.get("save_steps", 100),
            eval_strategy="steps",
            eval_steps=cfg.get("save_steps", 100),
            optim=cfg["optim"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            seed=cfg["seed"],
            output_dir=cfg["output_dir"],
            report_to="wandb",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=cfg["max_seq_length"],
            load_best_model_at_end=True,
        ),
    )

    try:
        print("[train] Starting training...")
        trainer.train()

        print(f"[train] Saving model to {cfg['output_dir']}/final...")
        trainer.save_model(f"{cfg['output_dir']}/final")
        processor.save_pretrained(f"{cfg['output_dir']}/final")

        # Log to MLflow
        mlflow.log_artifact(cfg["output_dir"], artifact_path="model")
        mlflow.log_metric("train_loss", trainer.state.log_history[-1].get("loss", 0))

        print(f"\n{'='*60}")
        print("  Training complete!")
        print(f"  Model saved → {cfg['output_dir']}/final")
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
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with tiny data subset to verify pipeline")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs: cfg["num_train_epochs"] = args.epochs
    if args.lr:     cfg["learning_rate"] = args.lr

    train(cfg, smoke_test=args.smoke_test)