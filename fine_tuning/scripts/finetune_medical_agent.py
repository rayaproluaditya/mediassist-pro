#!/usr/bin/env python3
"""
fine_tuning/scripts/finetune_medical_agent.py
==============================================
LoRA / QLoRA fine-tuning of TinyLlama-1.1B-Chat (or any Causal LM)
on the MediAssist Pro medical intake dataset.

Key hyperparameters:
  • Base model   : TinyLlama/TinyLlama-1.1B-Chat-v1.0
  • LoRA rank    : 16  (alpha=32)
  • Epochs       : 3
  • Batch size   : 4  (effective: 16 with grad_accum=4)
  • LR           : 2e-4 (cosine schedule)
  • Max seq len  : 512 tokens

Hardware:
  • GPU (CUDA) recommended; falls back to CPU for testing.
  • With bitsandbytes ≥ 0.41, 4-bit QLoRA reduces VRAM to ~6 GB on a T4.

Run:
    python fine_tuning/scripts/finetune_medical_agent.py \
        --train  fine_tuning/data/train.json \
        --val    fine_tuning/data/validation.json \
        --output fine_tuning/checkpoints/medical_agent_model

The adapter weights are saved to --output; load with PeftModel for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LEN  = 512
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
EPOCHS       = 3
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON structure in {path}")


def format_sample(sample: dict) -> str:
    """
    Convert an instruction-input-output dict into the TinyLlama chat template:
      <|system|>\n{instruction}<|user|>\n{input}<|assistant|>\n{output}
    """
    instruction = sample.get("instruction", "You are a medical intake assistant.")
    user_input  = sample.get("input", "").strip()
    output      = sample.get("output", "").strip()
    return (
        f"<|system|>\n{instruction}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n{output}"
    )


def build_hf_dataset(samples: list[dict]) -> Dataset:
    return Dataset.from_list([{"text": format_sample(s)} for s in samples])


# ── Fine-tune ─────────────────────────────────────────────────────────────────

def finetune(
    train_path: str,
    val_path: str,
    output_dir: str,
    base_model: str = BASE_MODEL,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    log.info("Base model: %s", base_model)

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    # ── Base model (no quantisation — works on CPU too) ───────────────────────
    log.info("Loading base model …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.config.use_cache       = False
    model.config.pretraining_tp  = 1

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Loading datasets …")
    train_ds = build_hf_dataset(load_json(train_path))
    val_ds   = build_hf_dataset(load_json(val_path))
    log.info("Train: %d  |  Val: %d", len(train_ds), len(val_ds))

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        gradient_checkpointing      = True,
        optim                       = "adamw_torch",
        learning_rate               = LR,
        weight_decay                = 0.001,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 50,
        fp16                        = (device == "cuda"),
        max_grad_norm               = 0.3,
        save_steps                  = 200,
        logging_steps               = 50,
        eval_strategy               = "steps",
        eval_steps                  = 200,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model            = model,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        args             = training_args,
        max_seq_length   = MAX_SEQ_LEN,
        dataset_text_field = "text",
        packing          = False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training …")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving adapter to %s …", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Log final eval loss ───────────────────────────────────────────────────
    metrics = trainer.evaluate()
    log.info("Final eval metrics: %s", metrics)

    loss_log_path = Path(output_dir) / "eval_metrics.json"
    loss_log_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved to %s", loss_log_path)
    log.info("✅ Fine-tuning complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MediAssist Pro LLM")
    parser.add_argument("--train",  default="fine_tuning/data/train.json")
    parser.add_argument("--val",    default="fine_tuning/data/validation.json")
    parser.add_argument("--output", default="fine_tuning/checkpoints/medical_agent_model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    finetune(
        train_path = args.train,
        val_path   = args.val,
        output_dir = args.output,
        base_model = args.base_model,
    )


if __name__ == "__main__":
    main()
