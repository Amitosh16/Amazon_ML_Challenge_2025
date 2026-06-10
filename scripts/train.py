#!/usr/bin/env python3
"""End-to-end training script for Mistral Quantile Regression."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to path so src/ imports work when called from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import StreamingTextDataset, collate_fn_factory, load_data, set_seed
from src.model import QuantileHead, QuantileModel
from src.loss import SmoothedPinballLoss
from src.trainer import Trainer


def setup_logging(save_dir: str):
    log_file = Path(save_dir) / "train.log"
    logger = logging.getLogger("mistral_qr")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def main():
    config = Config.from_args()
    logger = setup_logging(config.save_dir)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available — training will be extremely slow.")

    # Load data
    train_df, val_df, log_price_mean = load_data(config.train_csv, config.val_ratio, config.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    pad_id = tokenizer.pad_token_id or 0

    # Datasets / Loaders
    train_ds = StreamingTextDataset(train_df, tokenizer, config.max_length)
    val_ds = StreamingTextDataset(val_df, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn_factory(pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn_factory(pad_id),
    )

    # Base model + LoRA
    logger.info(f"Loading base model: {config.model_name}")
    base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_cfg).to(device)

    hidden_size = getattr(base.config, "hidden_size", getattr(base.config, "d_model", None))
    head = QuantileHead(hidden_size, config.k_quantiles, target_mean=log_price_mean).to(device, dtype=torch.bfloat16)
    model = QuantileModel(base, head)

    # Sanity check head initialization
    logger.info("Testing head initialization...")
    with torch.no_grad():
        dummy_ids = torch.randint(0, 1000, (2, 50)).to(device)
        dummy_mask = torch.ones_like(dummy_ids)
        q = model(dummy_ids, dummy_mask)
        logger.info(f"Init q range [{q.min():.2f}, {q.max():.2f}] spread={(q[:, -1] - q[:, 0]).mean():.2f}")

    # Loss and trainer
    taus = np.linspace(0.05, 0.95, config.k_quantiles)
    loss_fn = SmoothedPinballLoss(taus.tolist(), alpha=config.alpha_smoothing).to(device)
    trainer = Trainer(model, train_loader, val_loader, loss_fn, taus, config)

    if config.resume_from and Path(config.resume_from).exists():
        logger.info(f"Resuming from {config.resume_from}")
        model.load_state_dict(torch.load(config.resume_from, map_location=device))

    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
