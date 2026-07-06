# Predicting Customer Review Ratings

MSc thesis — Tilburg University, Data Science and Society, 2024.

Predicts a Google Maps review's star rating (1–5) from the review text and, where available, the business's managerial response. Benchmarks a Random Forest baseline against fine-tuned BERT and RoBERTa across two class-imbalance strategies (class weighting, random oversampling) and two feature sets (text only vs. text + response-derived features).

Full methodology: [`thesis/Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf`](thesis/Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf)

## Architecture

The transformer experiments use a fusion classification head that concatenates the `[CLS]` token embedding with optional tabular features (response timing, response length, personalization flag, price tier) before projecting to logits. Both BERT and RoBERTa are supported through a single `AutoModel`-based class. Class imbalance is handled either via per-class weighted cross-entropy loss or random oversampling of the training set. Hyperparameters were searched with Optuna, optimizing validation F1.

## Repository structure

```
src/
  data.py             load, split, balance, and combine text fields
  synthetic_data.py   generate structurally equivalent fake data for smoke testing
  model.py            FusionForSequenceClassification + WeightedLossTrainer
  baseline.py         Random Forest as an sklearn Pipeline
  train.py            unified training CLI
  tune.py             Optuna hyperparameter search
configs/
  best_hyperparams.json   best hyperparameters per experiment from the thesis search
results/
  metrics_thesis.csv      reported results from the thesis (Table 1)
  thesis_figures/         confusion matrices from the original thesis runs
thesis/
  *.pdf
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Smoke test — synthetic data, no checkpoint download required
python -m src.train --model rf --smoke-test
python -m src.train --model bert --balance oversample --features text_extra --smoke-test --epochs 1
python -m src.tune --model roberta --balance weight --smoke-test --trials 3

# Full run with real data
python -m src.train --model bert --balance oversample --features text_extra --data path/to/reviews.csv
python -m src.train --model roberta --balance weight --features text --data path/to/reviews.csv
python -m src.train --model rf --balance none --features text_extra --data path/to/reviews.csv

# Hyperparameter search
python -m src.tune --model bert --balance oversample --features text_extra --data path/to/reviews.csv --trials 30
```

Each run writes a confusion matrix to `results/figures/` and appends a metrics row to `results/metrics.csv`.

### CLI flags

| Flag | Options | Default |
|---|---|---|
| `--model` | `rf`, `bert`, `roberta` | required |
| `--balance` | `none`, `weight`, `oversample` | `none` |
| `--features` | `text`, `text_extra` | `text` |
| `--data` | path to CSV | required (or `--smoke-test`) |
| `--epochs` | int | `4` |
| `--batch-size` | int | `32` |
| `--learning-rate` | float | `3e-5` |
| `--dropout` | float | `0.1` |
| `--smoke-test` | flag | off |

### Expected CSV columns

`text`, `resp_text`, `rating`, `response_timing`, `response_length`, `general_personal`, `price_range`

## Results

Historical results are in `results/metrics_thesis.csv`. The original dataset is no longer available, so these numbers have not been reproduced by the code in this repository.

## Author

Ivaylo Papazov — Tilburg University, MSc Data Science and Society
