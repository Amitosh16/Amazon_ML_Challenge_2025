"""Configuration dataclass with CLI parsing support."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    # Paths
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_csv: str = "data/final_train.csv"
    save_dir: str = "checkpoints/default"
    resume_from: Optional[str] = None

    # Data
    val_ratio: float = 0.2
    max_length: int = 192
    seed: int = 42

    # Training
    batch_size: int = 2
    accum_steps: int = 8
    epochs: int = 4
    lr: float = 5e-6
    lr_head: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    early_stop_patience: int = 5

    # Quantile regression
    k_quantiles: int = 200
    alpha_smoothing: float = 1e-2
    eval_quantiles: Optional[List[float]] = field(default_factory=lambda: [0.25, 0.5, 0.75])

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.08

    # System
    eval_batch_size: int = 16
    num_workers: int = 2

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Mistral Quantile Regression Trainer")
        # Add every field as a CLI argument automatically
        for field_name, field_type in cls.__annotations__.items():
            if field_name == "eval_quantiles":
                parser.add_argument(f"--{field_name}", type=float, nargs="+", default=None)
            elif field_type == Optional[str]:
                parser.add_argument(f"--{field_name}", type=str, default=None)
            elif field_type == Optional[List[float]]:
                parser.add_argument(f"--{field_name}", type=float, nargs="+", default=None)
            elif field_type == bool:
                parser.add_argument(f"--{field_name}", type=lambda x: x.lower() == "true", default=getattr(cls, field_name))
            else:
                parser_type = str if field_type == str else field_type
                parser.add_argument(f"--{field_name}", type=parser_type, default=getattr(cls, field_name))
        args = parser.parse_args()
        return cls(**{k: v for k, v in vars(args).items() if v is not None})

    def __post_init__(self):
        if self.eval_quantiles is None:
            self.eval_quantiles = [0.25, 0.5, 0.75]
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
