import yaml
import mlflow
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from data_preprocessing import main as load_dataset

# Load config
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Load dataset in memory
train_dataset, val_dataset = load_dataset()

# Load pre-trained Med-GaMMa
model, processor = FastVisionModel.from_pretrained(
    cfg["pretrained_model"],
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

# Apply LoRA
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
    modules_to_save=["lm_head", "embed_tokens"]
)

# Data collator
data_collator = UnslothVisionDataCollator(model, processor, resize=cfg["resize"])

# Start MLflow run
mlflow.start_run()
mlflow.log_params(cfg)

# Trainer
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
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        output_dir=cfg["output_dir"],
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg["max_seq_length"]
    )
)

trainer.train()
mlflow.pytorch.log_model(model, "medgemma_pathvqa")
mlflow.end_run()