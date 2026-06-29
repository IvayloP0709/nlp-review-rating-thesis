"""Unified training entrypoint.

Replaces the 8 near-identical run scripts (one per model x balancing-strategy combination:
bert_sim.py, bert_bal.py, bert_overs.py, bert_add.py and their roBERTa equivalents) with one
parametrized pipeline:

    python -m src.train --model bert --balance oversample --features text_extra --data path.csv
    python -m src.train --model rf --balance weight --features text
    python -m src.train --model roberta --smoke-test   # synthetic data, no checkpoint download

`run_experiment` is the reusable core: src/tune.py's Optuna objective calls it directly with
trial-suggested hyperparameters, instead of duplicating the training loop per search script the
way bert_hyp_*.py / rob_hyp_*.py did.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, TrainingArguments

from src.baseline import build_rf_pipeline
from src.data import (
    EXTRA_FEATURE_COLUMNS,
    LABEL_TO_RATING,
    Splits,
    class_weights,
    combine_text_fields,
    load_preprocessed,
    make_splits,
    oversample,
)
from src.model import FusionForSequenceClassification, WeightedLossTrainer
from src.synthetic_data import make_synthetic_dataset

CHECKPOINTS = {"bert": "bert-base-uncased", "roberta": "roberta-base"}

# A tiny, randomly-initialized model config in place of a full pretrained checkpoint -- keeps
# --smoke-test fast and usable without network access to the full model weights.
SMOKE_TEST_CONFIG_OVERRIDES = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)


@dataclass
class ExperimentConfig:
    model: str  # "rf" | "bert" | "roberta"
    balance: str = "none"  # "none" | "weight" | "oversample"
    features: str = "text"  # "text" | "text_extra"
    data_path: Optional[str] = None
    learning_rate: float = 3e-5
    batch_size: int = 32
    epochs: int = 4
    dropout: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    rf_params: dict = field(default_factory=dict)  # overrides for build_rf_pipeline, used by src.tune
    smoke_test: bool = False
    seed: int = 42

    @property
    def run_name(self) -> str:
        return f"{self.model}_{self.balance}_{self.features}"

    @property
    def feature_columns(self) -> list[str]:
        return list(EXTRA_FEATURE_COLUMNS) if self.features == "text_extra" else []


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    val_f1_weighted: float
    accuracy: float
    f1_weighted: float
    confusion: np.ndarray = field(repr=False)


class ReviewDataset(Dataset):
    """Per-example dict of tensors, so extra_features always lines up with its own row's
    input_ids/labels -- the original bug was a *global* extra-features tensor sliced by batch
    position instead of indexed per example."""

    def __init__(self, encodings: dict, labels: torch.Tensor, extra_features: Optional[torch.Tensor] = None):
        self.encodings = encodings
        self.labels = labels
        self.extra_features = extra_features

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        if self.extra_features is not None:
            item["extra_features"] = self.extra_features[idx]
        return item


def _load_dataframe(config: ExperimentConfig) -> pd.DataFrame:
    if config.smoke_test or config.data_path is None:
        df = make_synthetic_dataset(n_rows=120, seed=config.seed)
        df["label"] = df["rating"].map({1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4})
        return df
    return load_preprocessed(config.data_path)


def _prepare_splits(config: ExperimentConfig) -> Splits:
    df = _load_dataframe(config)
    splits = make_splits(df, random_state=config.seed)
    for split in (splits.train, splits.val, splits.test):
        split["combined_text"] = combine_text_fields(split)
    return splits


def _maybe_oversample(train_df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    columns = ["combined_text", *feature_columns]
    X, y = oversample(train_df[columns], train_df["label"])
    return X.assign(label=y.to_numpy())


def _build_result(
    config: ExperimentConfig, val_f1_weighted: float, y_true: np.ndarray, y_pred: np.ndarray
) -> ExperimentResult:
    return ExperimentResult(
        config=config,
        val_f1_weighted=val_f1_weighted,
        accuracy=accuracy_score(y_true, y_pred),
        f1_weighted=f1_score(y_true, y_pred, average="weighted"),
        confusion=confusion_matrix(y_true, y_pred),
    )


def _run_rf(config: ExperimentConfig, splits: Splits) -> ExperimentResult:
    feature_columns = config.feature_columns
    train_df = splits.train
    if config.balance == "oversample":
        train_df = _maybe_oversample(train_df, feature_columns)
    class_weight = "balanced" if config.balance == "weight" else None

    pipeline = build_rf_pipeline(extra_feature_columns=feature_columns, class_weight=class_weight, **config.rf_params)
    columns = ["combined_text", *feature_columns]
    pipeline.fit(train_df[columns], train_df["label"])

    val_preds = pipeline.predict(splits.val[columns])
    val_f1 = f1_score(splits.val["label"], val_preds, average="weighted")

    test_preds = pipeline.predict(splits.test[columns])
    return _build_result(config, val_f1, splits.test["label"].to_numpy(), test_preds)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def _build_dataset(df: pd.DataFrame, tokenizer, feature_columns: Sequence[str]) -> ReviewDataset:
    encodings = tokenizer(df["combined_text"].tolist(), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    extra = torch.tensor(df[list(feature_columns)].to_numpy(), dtype=torch.float) if feature_columns else None
    return ReviewDataset(dict(encodings), labels, extra)


def _run_transformer(config: ExperimentConfig, splits: Splits) -> ExperimentResult:
    checkpoint = CHECKPOINTS[config.model]
    feature_columns = config.feature_columns

    train_df = splits.train
    class_weights_tensor = None
    if config.balance == "oversample":
        train_df = _maybe_oversample(train_df, feature_columns)
    elif config.balance == "weight":
        class_weights_tensor = class_weights(train_df["label"].to_numpy())

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dataset = _build_dataset(train_df, tokenizer, feature_columns)
    val_dataset = _build_dataset(splits.val, tokenizer, feature_columns)
    test_dataset = _build_dataset(splits.test, tokenizer, feature_columns)

    if config.smoke_test:
        model_config = AutoConfig.from_pretrained(checkpoint, **SMOKE_TEST_CONFIG_OVERRIDES)
        model = FusionForSequenceClassification(model_config, num_extra_features=len(feature_columns), dropout=config.dropout)
    else:
        model = FusionForSequenceClassification.from_checkpoint(
            checkpoint, num_extra_features=len(feature_columns), dropout=config.dropout
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            report_to=[],
            disable_tqdm=config.smoke_test,
            seed=config.seed,
        )
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=class_weights_tensor,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        val_metrics = trainer.evaluate()
        test_output = trainer.predict(test_dataset)

    preds = np.argmax(test_output.predictions, axis=-1)
    return _build_result(config, val_metrics["eval_f1_weighted"], test_output.label_ids, preds)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    torch.manual_seed(config.seed)
    splits = _prepare_splits(config)
    if config.model == "rf":
        return _run_rf(config, splits)
    return _run_transformer(config, splits)


def save_confusion_matrix(result: ExperimentResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{result.config.run_name}.png"
    labels = [int(LABEL_TO_RATING[i]) for i in range(result.confusion.shape[0])]
    plt.figure(figsize=(8, 6))
    sns.heatmap(result.confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted rating")
    plt.ylabel("True rating")
    plt.title(f"Confusion matrix: {result.config.run_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def append_metrics_row(result: ExperimentResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"
    row = {
        "run_name": result.config.run_name,
        "model": result.config.model,
        "balance": result.config.balance,
        "features": result.config.features,
        "val_f1_weighted": result.val_f1_weighted,
        "accuracy": result.accuracy,
        "f1_weighted": result.f1_weighted,
    }
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["run_name"] != row["run_name"]]
    else:
        existing = pd.DataFrame(columns=list(row.keys()))
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(path, index=False)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the review-rating baseline or transformer models.")
    parser.add_argument("--model", choices=["rf", "bert", "roberta"], required=True)
    parser.add_argument("--balance", choices=["none", "weight", "oversample"], default="none")
    parser.add_argument("--features", choices=["text", "text_extra"], default="text")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to the preprocessed CSV.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use synthetic data and (for bert/roberta) a tiny randomly-initialized model instead "
        "of a real checkpoint download, to verify the pipeline runs end to end.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> ExperimentResult:
    args = build_arg_parser().parse_args(argv)
    if args.data_path is None and not args.smoke_test:
        raise SystemExit("Pass --data <csv> or --smoke-test.")

    config = ExperimentConfig(
        model=args.model,
        balance=args.balance,
        features=args.features,
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    result = run_experiment(config)

    output_dir = Path(args.output_dir)
    figure_path = save_confusion_matrix(result, output_dir / "figures")
    metrics_path = append_metrics_row(result, output_dir)
    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "val_f1_weighted": result.val_f1_weighted,
                "accuracy": result.accuracy,
                "f1_weighted": result.f1_weighted,
            },
            indent=2,
        )
    )
    print(f"Confusion matrix written to {figure_path}")
    print(f"Metrics appended to {metrics_path}")
    return result


if __name__ == "__main__":
    main()
