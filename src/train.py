"""
Usage:
    python -m src.train
    python -m src.train --smoke-test
    python -m src.train --epochs 5 --lr 1e-4
"""

import os
import sys
import argparse
import warnings
import logging
import contextlib
import io as _io
import yaml
import torch
import wandb
from huggingface_hub import login
from torch.nn.utils.rnn import pad_sequence

# suppress processor deprecation warnings that flood stdout during training
warnings.filterwarnings("ignore", message=".*processor.image_token.*")
warnings.filterwarnings("ignore", message=".*boi_token.*")
warnings.filterwarnings("ignore", message=".*use_cache.*gradient.*")
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")

class _DeprecationFilter(logging.Filter):
    _SKIP = ("processor.image_token", "boi_token", "use_cache", "tie_word_embeddings")
    def filter(self, record):
        return not any(kw in record.getMessage() for kw in self._SKIP)

logging.getLogger("transformers").addFilter(_DeprecationFilter())

# peft pulls in bitsandbytes even when 4-bit is disabled; silence the import
# error on CUDA 13.0 where the native lib is missing (harmless when 4-bit is off)
with contextlib.redirect_stderr(_io.StringIO()):
    try:
        import bitsandbytes as _bnb  # noqa: F401
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import main as load_dataset


def _log_gpu():
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    resv = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {alloc:.1f} GB used | {resv:.1f} GB reserved | {total:.1f} GB total")


def get_collate_fn(processor, max_seq_length: int):
    def collate_fn(examples):
        batch_input_ids = []
        batch_attention = []
        batch_pixel_values = []
        batch_prompt_lens = []

        for e in examples:
            img = None
            for msg in e["messages"]:
                for part in msg["content"]:
                    if part["type"] == "image":
                        img = part["image"]
                        break
                if img:
                    break

            full_text = processor.apply_chat_template(
                e["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            full_inputs = processor(
                text=full_text,
                images=img,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )

            # add_generation_prompt=True gives everything up to the start of the
            # assistant turn — used to compute prompt length for label masking
            prompt_msgs = [m for m in e["messages"] if m["role"] != "assistant"]
            prompt_text = processor.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_inputs = processor(
                text=prompt_text,
                images=img,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )

            batch_input_ids.append(full_inputs["input_ids"][0])
            batch_attention.append(full_inputs["attention_mask"][0])
            batch_pixel_values.append(full_inputs["pixel_values"][0])
            batch_prompt_lens.append(prompt_inputs["input_ids"].shape[1])

        pad_id = processor.tokenizer.pad_token_id
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(batch_attention, batch_first=True, padding_value=0)
        pixel_values = torch.stack(batch_pixel_values)

        # loss computed only on assistant response — mask pad tokens and the
        # entire instruction/question prefix so the model isn't rewarded for
        # memorising the system prompt
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        for i, prompt_len in enumerate(batch_prompt_lens):
            labels[i, :prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            # token_type_ids omitted — Gemma is decoder-only
        }

    return collate_fn


def train(cfg: dict, smoke_test: bool = False):
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    model_name = cfg.get("pretrained_model", "google/medgemma-4b-it")
    use_fa2 = cfg.get("use_flash_attention", False)
    use_dora = cfg.get("use_dora", False)
    use_rslora = cfg.get("use_rslora", False)
    # sdpa = PyTorch built-in fused attention, no extra install needed
    attn_impl = cfg.get("attn_implementation", "flash_attention_2" if use_fa2 else "sdpa")
    load_in_4bit = cfg.get("load_in_4bit", False)

    print("\n" + "=" * 64)
    print("  Path-VQA Med-GaMMa Fine-Tuning")
    print("=" * 64)
    print(f"  Model       : {model_name}")
    print(f"  Dataset     : {cfg['dataset_name']}")
    print(f"  Epochs      : {cfg['num_train_epochs']}")
    print(f"  Batch       : {cfg['per_device_train_batch_size']} × {cfg['gradient_accumulation_steps']} accum "
          f"= {cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']} effective")
    print(f"  LR          : {cfg['learning_rate']}")
    print(f"  LoRA r      : {cfg['lora_r']}  alpha={cfg['lora_alpha']}")
    print(f"  DoRA        : {use_dora}")
    print(f"  RSLoRA      : {use_rslora}")
    print(f"  Attention   : {attn_impl}")
    print(f"  4-bit       : {load_in_4bit}")
    print(f"  Smoke test  : {smoke_test}")
    print(f"  Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 64 + "\n")

    hf_token = cfg.get("hf_token") or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[auth] HuggingFace login OK")
    else:
        raise ValueError(
            "HF_TOKEN not found. Set it via env var or add hf_token to config."
        )

    train_dataset, val_dataset = load_dataset(
        dataset_name=cfg["dataset_name"],
        use_enhanced=cfg.get("use_enhanced_answer", True),
        max_train_samples=50 if smoke_test else None,
        max_val_samples=20 if smoke_test else None,
    )

    compute_dtype = torch.bfloat16

    if load_in_4bit:
        with contextlib.redirect_stderr(_io.StringIO()):
            from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        )
    else:
        bnb_config = None

    quant_label = "4-bit NF4, double_quant" if load_in_4bit else "bfloat16"
    print(f"[model] Loading {model_name} ({quant_label}, attn={attn_impl})...")
    load_kwargs = dict(device_map="auto", attn_implementation=attn_impl)
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = compute_dtype

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        if use_fa2:
            print("[model] Flash Attention 2 active")
    except Exception as exc:
        if use_fa2:
            print(f"[warn] Flash Attention 2 unavailable ({type(exc).__name__}: {exc})")
            print("[warn] Falling back to eager attention")
            load_kwargs["attn_implementation"] = "eager"
            model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        else:
            raise

    processor = AutoProcessor.from_pretrained(model_name)
    _log_gpu()

    # modules_to_save intentionally omitted — adding lm_head + embed_tokens
    # pushes trainable params to ~1.38B and OOMs on the optimizer state
    adapter_mode = "DoRA" if use_dora else "LoRA"
    print(f"[model] Attaching {adapter_mode} adapters "
          f"(r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, "
          f"rslora={use_rslora})...")
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        use_dora=use_dora,
        use_rslora=use_rslora,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    _log_gpu()

    collate_fn = get_collate_fn(processor, cfg["max_seq_length"])

    run_name = (
        f"medgemma_{'dora' if use_dora else 'lora'}"
        f"_r{cfg['lora_r']}"
        f"_fa2{int(use_fa2)}"
        f"_{'smoke' if smoke_test else 'full'}"
    )
    wandb.init(
        project=cfg.get("wandb_project", "path-vqa-medgemma"),
        entity=cfg.get("wandb_entity") or None,
        name=run_name,
        config={
            **cfg,
            "smoke_test": smoke_test,
            "use_flash_attention": use_fa2,
            "use_dora": use_dora,
            "use_rslora": use_rslora,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        dir=os.environ.get("WANDB_DIR", "wandb"),
    )

    # warmup_ratio deprecated in transformers v5.2 — compute steps directly
    if smoke_test:
        warmup_steps = 1
    else:
        effective_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
        total_steps = max(1, len(train_dataset) // effective_batch) * cfg["num_train_epochs"]
        warmup_steps = max(1, int(total_steps * cfg.get("warmup_ratio", 0.05)))

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=1 if smoke_test else cfg["num_train_epochs"],
        max_steps=5 if smoke_test else -1,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        warmup_steps=warmup_steps,
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        optim=cfg.get("optim", "adamw_torch_fused"),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        push_to_hub=bool(cfg.get("hub_model_id")),
        hub_model_id=cfg.get("hub_model_id"),
        eval_strategy="steps",
        eval_steps=cfg.get("save_steps", 100),
        load_best_model_at_end=True,
        seed=cfg.get("seed", 3407),
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    try:
        torch.cuda.empty_cache()
        print("[train] Starting training...\n")
        trainer.train()

        out_final = f"{cfg['output_dir']}/final"
        print(f"\n[train] Saving model → {out_final}")
        trainer.save_model(out_final)
        processor.save_pretrained(out_final)

        if not smoke_test:
            artifact_name = (
                cfg.get("hub_model_id", "medgemma-path-vqa")
                .split("/")[-1]
                .replace(" ", "-")
            )
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=(
                    f"Med-GaMMa 4B fine-tuned on PathVQA Enhanced. "
                    f"{'DoRA' if use_dora else 'LoRA'} r={cfg['lora_r']}, "
                    f"FA2={use_fa2}"
                ),
                metadata={
                    "base_model": model_name,
                    "dataset": cfg["dataset_name"],
                    "lora_r": cfg["lora_r"],
                    "lora_alpha": cfg["lora_alpha"],
                    "use_dora": use_dora,
                    "use_rslora": use_rslora,
                    "use_flash_attention": use_fa2,
                    "epochs": cfg["num_train_epochs"],
                    "learning_rate": cfg["learning_rate"],
                    "batch_effective": (
                        cfg["per_device_train_batch_size"]
                        * cfg["gradient_accumulation_steps"]
                    ),
                    "hub_model_id": cfg.get("hub_model_id"),
                },
            )
            artifact.add_dir(out_final, name="adapter")
            wandb.log_artifact(artifact)
            print(f"[wandb] Model artifact logged → {artifact_name}:latest")

        print(f"\n{'='*64}")
        print("  Training complete!")
        print(f"  Model saved  → {out_final}")
        if cfg.get("hub_model_id"):
            print(f"  HF Hub push  → {cfg['hub_model_id']}")
        print(f"{'='*64}\n")
        _log_gpu()

    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving checkpoint...")
        trainer.save_model(f"{cfg['output_dir']}/interrupted")
        processor.save_pretrained(f"{cfg['output_dir']}/interrupted")

    finally:
        wandb.finish()

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Med-GaMMa on PathVQA Enhanced")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5 steps on 50/20 samples to verify the full pipeline")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["num_train_epochs"] = args.epochs
    if args.lr:
        cfg["learning_rate"] = args.lr

    train(cfg, smoke_test=args.smoke_test)
