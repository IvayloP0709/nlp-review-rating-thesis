"""Unified Optuna hyperparameter search.

Replaces the 7 near-identical hyperparameter-search scripts (bert_hyp_.py, bert_hyp_add.py,
bert_hyp_overs.py, bert_hyp_weight.py, rob_hyp_.py, rob_hyp_overs.py, rob_hyp_weight.py) with
one CLI that calls src.train.run_experiment as the Optuna objective, rather than duplicating
the training loop in every search script:

    python -m src.tune --model bert --balance oversample --features text --trials 10
    python -m src.tune --model rf --balance weight --trials 10

Search ranges match the originals: random search (RandomSampler), maximizing *validation*
weighted F1 -- the original scripts tuned on validation and only the final src.train run
reports test-set numbers, to avoid tuning against the test set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Optional, Sequence

import optuna

from src.train import ExperimentConfig, run_experiment


def _suggest_transformer_config(trial: optuna.Trial, base: ExperimentConfig) -> ExperimentConfig:
    return replace(
        base,
        learning_rate=trial.suggest_categorical("learning_rate", [2e-5, 3e-5, 5e-5]),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        epochs=trial.suggest_int("epochs", 2, 4),
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        gradient_accumulation_steps=trial.suggest_int("gradient_accumulation_steps", 1, 4),
    )


def _suggest_rf_config(trial: optuna.Trial, base: ExperimentConfig) -> ExperimentConfig:
    rf_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 2000, step=200),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        max_samples=trial.suggest_categorical("max_samples", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        min_samples_leaf=trial.suggest_categorical("min_samples_leaf", [1, 5, 10, 15, 20]),
    )
    return replace(base, rf_params=rf_params)


def make_objective(base_config: ExperimentConfig):
    suggest = _suggest_rf_config if base_config.model == "rf" else _suggest_transformer_config

    def objective(trial: optuna.Trial) -> float:
        config = suggest(trial, base_config)
        result = run_experiment(config)
        trial.set_user_attr("test_f1_weighted", result.f1_weighted)
        trial.set_user_attr("test_accuracy", result.accuracy)
        return result.val_f1_weighted

    return objective


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["rf", "bert", "roberta"], required=True)
    parser.add_argument("--balance", choices=["none", "weight", "oversample"], default="none")
    parser.add_argument("--features", choices=["text", "text_extra"], default="text")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to the preprocessed CSV.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use synthetic data and (for bert/roberta) a tiny randomly-initialized model instead "
        "of a real checkpoint download, to verify the search runs end to end.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> optuna.Study:
    args = build_arg_parser().parse_args(argv)
    if args.data_path is None and not args.smoke_test:
        raise SystemExit("Pass --data <csv> or --smoke-test.")

    base_config = ExperimentConfig(
        model=args.model,
        balance=args.balance,
        features=args.features,
        data_path=args.data_path,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=args.seed))
    study.optimize(make_objective(base_config), n_trials=args.trials)

    best_trial = study.best_trial
    print(
        json.dumps(
            {
                "best_val_f1_weighted": study.best_value,
                "best_params": study.best_params,
                "best_trial_test_f1_weighted": best_trial.user_attrs.get("test_f1_weighted"),
                "best_trial_test_accuracy": best_trial.user_attrs.get("test_accuracy"),
            },
            indent=2,
        )
    )
    return study


if __name__ == "__main__":
    main()
