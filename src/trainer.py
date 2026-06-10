"""Training loop with validation, checkpointing, and early stopping."""

import gc
import logging
import math

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import Config

logger = logging.getLogger("mistral_qr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, taus, config: Config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.taus = taus
        self.config = config
        self.best_smape = float("inf")
        self.patience = 0
        self.epoch = 0

        # Separate LR for head vs base
        head_params = [p for n, p in model.named_parameters() if "head" in n]
        base_params = [p for n, p in model.named_parameters() if "head" not in n]
        self.opt = torch.optim.AdamW(
            [
                {"params": base_params, "lr": config.lr},
                {"params": head_params, "lr": config.lr_head},
            ],
            weight_decay=config.weight_decay,
        )

        total_steps = math.ceil(len(train_loader) / config.accum_steps) * config.epochs
        warmup = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(self.opt, warmup, total_steps)

        self.median_idx = int(np.argmin(np.abs(taus - 0.5)))
        self.best_ckpt = f"{config.save_dir}/best.pt"
        self.last_ckpt = f"{config.save_dir}/last.pt"

    @torch.no_grad()
    def evaluate(self):
        """Compute validation SMAPE and MAPE on the median prediction."""
        self.model.eval()
        total_smape = total_mape = total_n = 0
        for ids, mask, y in self.val_loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            with autocast(device_type=str(DEVICE), dtype=torch.bfloat16):
                q = self.model(ids, mask)
            preds = torch.expm1(q[:, self.median_idx])
            targets = torch.expm1(y)
            denom = (preds.abs() + targets.abs()) / 2 + 1e-8
            total_smape += (preds - targets).abs().div(denom).sum().item()
            total_mape += (preds - targets).abs().div(targets.abs() + 1e-8).sum().item()
            total_n += len(y)
        if total_n == 0:
            return float("inf"), float("inf")
        return (total_smape / total_n) * 100, (total_mape / total_n) * 100

    def train(self):
        """Main training loop."""
        for epoch in range(self.config.epochs):
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            running_loss, steps = 0.0, 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            for i, (ids, mask, y) in enumerate(pbar):
                ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

                with autocast(device_type=str(DEVICE), dtype=torch.bfloat16):
                    q = self.model(ids, mask)
                    loss = self.loss_fn(q, y) / self.config.accum_steps

                loss.backward()
                running_loss += loss.item()
                steps += 1

                if (i + 1) % self.config.accum_steps == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad(set_to_none=True)

                if i % 1000 == 0:
                    spread = (q[:, -1] - q[:, 0]).mean().item()
                    mae = (y - q[:, self.median_idx]).abs().mean().item()
                    logger.info(
                        f"[Epoch {epoch + 1}|Step {i}] "
                        f"Loss={running_loss / max(1, steps):.4f} "
                        f"Spread={spread:.2f} MAE={mae:.2f}"
                    )

                pbar.set_postfix(loss=f"{running_loss / max(1, steps):.4f}")

            avg_loss = running_loss / steps
            val_smape, val_mape = self.evaluate()
            logger.info(
                f"Epoch {epoch + 1} done | TrainLoss={avg_loss:.4f} | "
                f"Val SMAPE={val_smape:.2f}% | Val MAPE={val_mape:.2f}%"
            )

            # Early stopping and checkpoints
            if val_smape < self.best_smape:
                self.best_smape = val_smape
                self.patience = 0
                torch.save(self.model.state_dict(), self.best_ckpt)
                logger.info(f"Saved new best model (SMAPE={val_smape:.2f})")
            else:
                self.patience += 1
                torch.save(self.model.state_dict(), self.last_ckpt)

            if self.patience >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
