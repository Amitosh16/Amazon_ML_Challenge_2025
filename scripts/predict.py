#!/usr/bin/env python3
"""Inference script: load a consolidated checkpoint and predict on test.csv."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class QuantileHead(nn.Module):
    def __init__(self, hidden_size: int, k: int = 200):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, k),
        )

    def forward(self, x):
        raw = self.net(x)
        first_q = raw[:, 0:1]
        deltas = torch.nn.functional.softplus(raw[:, 1:])
        z = torch.cat([first_q, deltas], dim=1)
        return torch.cumsum(z, dim=1)


class QuantileModel(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base, self.head = base, head

    def forward(self, ids, mask):
        out = self.base(ids, attention_mask=mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        seq_lens = mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(ids.size(0), device=ids.device), seq_lens]
        return self.head(pooled)


def parse_args():
    parser = argparse.ArgumentParser(description="Mistral Quantile Regression Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--test_csv", required=True, help="Path to test.csv")
    parser.add_argument("--output_csv", required=True, help="Where to save predictions")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--k_quantiles", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model skeleton...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.08,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    hidden_size = getattr(peft_model.config, "hidden_size", 4096)
    head = QuantileHead(hidden_size, k=args.k_quantiles)
    qm = QuantileModel(peft_model, head)

    print(f"Loading consolidated checkpoint from {args.checkpoint}...")
    qm.load_state_dict(torch.load(args.checkpoint, map_location=device))
    qm.to(device)
    qm.eval()
    print("Model loaded successfully.")

    # Median index
    taus = np.linspace(0.05, 0.95, args.k_quantiles)
    median_idx = int(np.argmin(np.abs(taus - 0.5)))
    print(f"Using index {median_idx} for the median prediction.")

    test = pd.read_csv(args.test_csv)
    print(f"Loaded test set: {len(test)} rows")

    prices = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Predicting"):
        text = str(row["catalog_content"])
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=args.max_length
        ).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=str(device), dtype=torch.bfloat16):
            out = qm(enc["input_ids"], enc["attention_mask"])
            pred_log_price = out[0, median_idx].cpu().item()
        price = np.expm1(pred_log_price)
        prices.append(float(price))

    test["price"] = prices
    output_df = test[["sample_id", "price"]]
    output_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to: {args.output_csv}")
    print(output_df.head())


if __name__ == "__main__":
    main()
