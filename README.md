# Amazon ML Challenge 2025 — Top 30 (Text-only Quantile Regression)

> **Competition:** Amazon ML Challenge India 2025  
> **Result:** **Top 30 / ~80,000 participants**  
> **Approach:** Single text-only model (no images) using Mistral-7B + LoRA + custom quantile regression head

---

# Mistral Quantile Regression for Price Prediction

A structured, production-ready PyTorch project for fine-tuning **Mistral-7B-Instruct-v0.2** with **LoRA** to predict product prices via **quantile regression**. Instead of a single point estimate, the model learns a full conditional distribution over prices by predicting 200 quantile levels, allowing richer uncertainty quantification.

This repository contains the exact code that achieved a **Top 30 finish out of approximately 80,000 teams** in the **Amazon ML Challenge India 2025**, demonstrating that a carefully tuned language model with quantile regression can compete at the highest level—even without using image data.

---

## What This Project Does

- **Task:** Predict the price of a product given its textual catalog description (`catalog_content`).
- **Approach:** Fine-tune a large language model (Mistral-7B) with parameter-efficient LoRA adapters and a custom quantile regression head.
- **Output:** 200 quantile estimates. The median (50th percentile) is used as the final point prediction.
- **Target Variable:** `price`, log-transformed (`log1p`) during training and exponentiated (`expm1`) at inference.

---

## Competition Context

This solution was built for the **Amazon ML Challenge India 2025**, where the task was to predict the price of products sold on Amazon based on catalog metadata. The dataset contained:

- `sample_id` — unique product identifier
- `catalog_content` — textual product description/title
- `image_link` — product image URL (this solution is **text-only**)
- `price` — target variable (product selling price)

**Why this is impressive:**
- ~80,000 teams participated.
- Most top teams used ensemble-heavy pipelines combining vision models (CNNs/ViTs) + text models + gradient boosting.
- This single-model, text-only approach reached the **Top 30** by treating price prediction as a **quantile regression problem** on a large language model, avoiding the complexity and inference cost of multi-modal ensembles.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Base Model** | `mistralai/Mistral-7B-Instruct-v0.2` |
| **PEFT** | LoRA (`r=64`, `alpha=128`) on attention and MLP layers |
| **Quantile Head** | Custom MLP head that outputs 200 quantiles via cumulative sum of softplus deltas |
| **Loss** | Smoothed pinball loss (quantile loss) with automatic scaling |
| **Optimization** | AdamW with different learning rates for base (`5e-6`) and head (`2e-5`) |
| **Training Tricks** | Gradient accumulation (8 steps), gradient checkpointing, mixed precision (`bfloat16`), gradient clipping |
| **Validation** | SMAPE and MAPE on the median prediction |
| **Checkpointing** | Saves `best.pt` (lowest Val SMAPE) and `last.pt`; early stopping with patience=5 |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/                     # Place your CSVs here
├── notebooks/
│   └── Quantile Regression.ipynb   # Original Colab notebook
├── src/
│   ├── config.py             # Hyperparameters and paths
│   ├── data.py               # Dataset, collate function, data loading
│   ├── model.py              # QuantileHead + QuantileModel wrapper
│   ├── loss.py               # SmoothedPinballLoss
│   └── trainer.py            # Training loop with evaluation & early stopping
└── scripts/
    ├── train.py              # End-to-end training script
    └── predict.py            # Inference on test.csv
```

---

## Dataset Format

Your CSV must contain at least these columns:

| Column | Type | Description |
|--------|------|-------------|
| `catalog_content` | string | Product description / catalog text |
| `price` | float | Product price (must be > 0) |

The training script will:
1. Automatically detect the price column (case-insensitive).
2. Drop rows with missing/invalid prices.
3. Apply `log1p(price)` to stabilize variance.

For inference, `test.csv` only needs `sample_id` and `catalog_content`.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training

```bash
python scripts/train.py \
    --train_csv data/final_train.csv \
    --save_dir checkpoints/run_1 \
    --epochs 4 \
    --batch_size 2 \
    --accum_steps 8
```

All hyperparameters can be overridden via CLI. Run `python scripts/train.py --help` for the full list.

**Key CLI arguments:**
- `--train_csv` – path to training CSV
- `--save_dir` – directory to save checkpoints and logs
- `--resume_from` – path to a checkpoint to resume from
- `--epochs`, `--batch_size`, `--accum_steps`, `--lr`, `--lr_head`
- `--lora_r`, `--lora_alpha`, `--lora_dropout`
- `--k_quantiles` – number of quantile outputs (default 200)

### 3. Inference

```bash
python scripts/predict.py \
    --checkpoint checkpoints/run_1/best.pt \
    --test_csv data/test.csv \
    --output_csv data/test_predictions.csv
```

This loads the consolidated `best.pt` (which includes both LoRA adapters and the quantile head), reconstructs the architecture, and writes a CSV with `sample_id` and `price`.

---

## Model Architecture Details

### Quantile Head

Instead of predicting a single value, the head predicts:
- `q[0]` – the first quantile value (raw).
- `delta[1:]` – positive increments via `softplus`.
- Final quantiles: `Q = cumsum([q[0], delta[1], ..., delta[k-1]])`.

This guarantees monotonically increasing quantiles without extra constraints.

### Initialization Trick

The bias of the final layer is initialized so that the total spread of the 200 quantiles roughly matches `3 × mean(log_price)`, clamped to `[4, 10]`. This prevents collapse at the start of training.

---

## Expected Results

Based on the original notebook run (4 epochs, ~60k training samples, A100 GPU) during the Amazon ML Challenge 2025:

| Epoch | Train Loss | Val SMAPE | Val MAPE |
|-------|------------|-----------|----------|
| 1 | 0.0150 | 46.31% | 64.10% |
| 2 | 0.0139 | 43.83% | 72.40% |
| 3 | 0.0133 | 42.70% | 63.29% |
| 4 | 0.0128 | **42.66%** | 63.53% |

> These are **SMAPE/MAPE on price** (not log-price), so values in the 40s are typical for this dataset and task. Lower is better.

### Competition Achievement

- **Final Rank:** Top 30 / ~80,000 participating teams  
- **Model Count:** Single model (text-only)  
- **Key Insight:** Treating price prediction as quantile regression on a language model provided competitive accuracy without the cost and complexity of vision models or large ensembles.

---

## Hardware Notes

- **GPU:** The original notebook used an NVIDIA A100. Training on smaller GPUs is possible with smaller `batch_size` / larger `accum_steps`, but Mistral-7B in `bfloat16` still needs ~16–20 GB of VRAM.
- **CPU:** Not recommended; training would be impractically slow.

---

## Extending the Project

- **Different LLM:** Change `model_name` in `src/config.py` or via `--model_name`.
- **More/Fewer Quantiles:** Adjust `--k_quantiles`. Remember to update `median_idx` logic if you change the tau range.
- **Custom Metrics:** Modify `Trainer.evaluate()` in `src/trainer.py`.
- **W&B / TensorBoard:** Add a callback in `Trainer.train()`.

---

## License

This project is provided as-is for research and educational purposes. Please respect the licenses of the underlying models (Mistral-7B) and libraries (Hugging Face, PEFT, etc.).

## Acknowledgments

- Original notebook: *Mistral Quantile Regression Trainer – Final Version with Auto-Scaled Initialization, Diagnostics, Validation SMAPE/MAPE, and Checkpointing*.
- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), and [PyTorch](https://pytorch.org/).
