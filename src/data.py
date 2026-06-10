"""Dataset and data loading utilities."""

import logging
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger("mistral_qr")


class StreamingTextDataset(Dataset):
    """PyTorch Dataset that streams rows from a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["catalog_content"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(row["log_price"], dtype=torch.float32),
        )


def collate_fn_factory(pad_id: int):
    """Build a collate function for dynamic padding."""

    def collate_fn(batch):
        input_ids = [b[0] for b in batch]
        attn = [b[1] for b in batch]
        y = torch.stack([b[2] for b in batch])
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attn_padded = pad_sequence(attn, batch_first=True, padding_value=0)
        return input_ids_padded, attn_padded, y

    return collate_fn


def load_data(csv_path: str, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load CSV, clean, log-transform price, and split train/val."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    if price_col is None:
        raise ValueError("No price column found in the CSV.")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["catalog_content", price_col])
    df = df[df[price_col] > 0]
    df["log_price"] = np.log1p(df[price_col])
    df = df[np.isfinite(df["log_price"])]

    logger.info(f"Log price range: [{df['log_price'].min():.2f}, {df['log_price'].max():.2f}]")
    log_price_mean = df["log_price"].mean()
    logger.info(f"Log price mean={log_price_mean:.2f}")

    train_df, val_df = train_test_split(
        df[["catalog_content", "log_price"]], test_size=val_ratio, random_state=seed
    )
    return train_df, val_df, log_price_mean


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
