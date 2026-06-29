# Predicting Customer Review Ratings

**Tilburg University · MSc Data Science and Society · Master's Thesis, 2024**

Predicts a customer review's star rating (1-5) from the review text and, where present, the
business's managerial response to it. Compares a Random Forest baseline against fine-tuned
BERT and RoBERTa, two class-imbalance strategies (class weighting, random oversampling), and an
optional fusion of four response-derived features (timing, length, personalization, business
price tier) into the transformer's classification head.

Full methodology and discussion: [`thesis/Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf`](thesis/Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf).

## Layout

```
src/
  data.py            # split / balance / text-combining logic shared by every experiment
  synthetic_data.py   # fabricates data in the same shape, for running the pipeline without the real dataset
  model.py             # transformer + optional feature-fusion head (works with any AutoModel checkpoint)
  baseline.py           # Random Forest as an sklearn Pipeline
  train.py               # python -m src.train --model {rf,bert,roberta} --balance {none,weight,oversample} --features {text,text_extra}
  tune.py                 # python -m src.tune   (same flags) -- Optuna search over the same training function
configs/
  best_hyperparams.json  # best hyperparameters per experiment, as found by the thesis's search
results/
  metrics_thesis.csv      # Table 1 from the thesis (historical -- see "On the results" below)
  thesis_figures/          # confusion matrices from the original thesis runs
thesis/
  *.pdf
```

## On the results

The original dataset (Google Local Data, via Yan et al. 2023 and Li et al. 2022, merged and
feature-engineered per the thesis's Section 4.3) is **no longer available** -- it lived on Tilburg/
Colab storage that's since been cleared. `results/metrics_thesis.csv` and `results/thesis_figures/`
are the thesis's originally reported numbers, kept for reference; they have **not** been
reproduced by the code in this repository.

The code itself was rewritten from the original scripts into the `src/` pipeline below, including
three bugs found while doing that (see next section). Rerunning the rewritten pipeline against the
real data would likely produce different numbers than `metrics_thesis.csv` -- probably more
trustworthy ones, but unverified, since that data is gone. To confirm the pipeline is at least
mechanically correct without it, `src/synthetic_data.py` fabricates structurally-equivalent fake
data (same columns, same missing-response rate, same class imbalance) so every model/balance/
feature combination can be run end to end via `--smoke-test`; metrics from that data carry no real
signal by construction.

## Bugs found while rewriting this

1. **Feature-fusion misalignment.** The original "additional features" experiments built one
   tensor of response-derived features over the *entire* dataset in its original row order, then
   every training batch did `extra_features[:len(input_ids)]` -- always the first `batch_size`
   rows, never the rows actually in that (shuffled) batch. The reported best model (BERT +
   oversampling + extra features) was trained with tabular features that didn't correspond to the
   text they were fused with. Fixed in `src/model.py` / `src/train.py`: the features are now a
   column of the same per-example batch as the text.
2. **Class-weight indexing.** `compute_class_weights()` in the "weight balanced" experiments
   returned a *per-sample* tensor (one weight per training row), then the caller sliced its first
   5 entries by position and passed that to `CrossEntropyLoss(weight=...)` as if it were "the
   weight for class 0..4." It was actually the weight of the first 5 training rows. Fixed in
   `src/data.py::class_weights()`, which now returns one weight per class, indexed by class id.
3. **Silent "nan" injection.** `resp_text` (the managerial response) is null for roughly 89% of
   rows (the original RF baseline notebook's `df.info()` output showed 1,510 of 14,075 rows with a
   non-null `resp_text`, before that notebook was removed in this cleanup). Every script built the model
   input with `str(value)`, and `str(float('nan'))` is the literal string `"nan"` -- so most
   training examples, across every experiment including the plain-text baseline, had the substring
   "nan" appended to their review text. Fixed in `src/data.py::combine_text_fields()`, which uses
   `fillna("")` instead.

None of these were caught by the thesis's evaluation at the time; all three affect the literal
input/loss the models were trained on, not just hyperparameters.

## Running it

```bash
pip install -r requirements.txt

# Verify the pipeline runs end to end (synthetic data, no checkpoint download for bert/roberta):
python -m src.train --model rf --smoke-test
python -m src.train --model bert --balance oversample --features text_extra --smoke-test --epochs 1
python -m src.tune --model roberta --balance weight --smoke-test --trials 3

# With the real preprocessed CSV (text, resp_text, rating, response_timing, response_length,
# general_personal, price_range -- see src/data.py):
python -m src.train --model bert --balance oversample --features text_extra --data path/to/reviews.csv
```

Each run writes a confusion matrix to `results/figures/` and appends a row to `results/metrics.csv`
(both gitignored -- they're run-local artifacts, not the historical numbers above).

## Author

**Ivaylo Papazov** -- Tilburg University, MSc Data Science and Society

Academic work; please cite if used.
