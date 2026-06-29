"""Fabricate a CSV in the post-preprocessing shape the pipeline expects.

The real dataset (Yan et al. 2023 / Li et al. 2022 Google Local Data, merged and feature-
engineered per the thesis) is not redistributable and is no longer available to the author.
This generates structurally-equivalent fake data -- including a realistic missing-response
rate -- purely so the rest of the pipeline can be smoke-tested end to end. Do not read
anything into metrics produced from this data; it carries no real signal by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_POSITIVE_WORDS = ["great", "friendly", "fast", "clean", "delicious", "helpful"]
_NEGATIVE_WORDS = ["slow", "rude", "overpriced", "dirty", "late", "disappointing"]
_RATING_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
_RATING_PROBS = [0.05, 0.05, 0.15, 0.30, 0.45]  # mirrors the thesis's 5-star-heavy imbalance


def make_synthetic_dataset(n_rows: int = 200, seed: int = 42, response_rate: float = 0.11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ratings = rng.choice(_RATING_VALUES, size=n_rows, p=_RATING_PROBS)

    def review_text(rating: float) -> str:
        vocab = _NEGATIVE_WORDS if rating <= 2 else _POSITIVE_WORDS
        words = rng.choice(vocab, size=rng.integers(4, 12))
        return "The service was " + " ".join(words) + "."

    has_response = rng.random(n_rows) < response_rate
    resp_texts = [
        "Thank you for your feedback, we appreciate it." if has else np.nan for has in has_response
    ]

    return pd.DataFrame(
        {
            "text": [review_text(r) for r in ratings],
            "resp_text": resp_texts,
            "rating": ratings,
            "response_timing": rng.integers(0, 60, size=n_rows),
            "response_length": [len(str(r).split()) if has else 0 for r, has in zip(resp_texts, has_response)],
            "general_personal": rng.integers(0, 2, size=n_rows),
            "price_range": rng.integers(1, 5, size=n_rows),
        }
    )


def write_synthetic_csv(path: str, n_rows: int = 200, seed: int = 42) -> None:
    make_synthetic_dataset(n_rows=n_rows, seed=seed).to_csv(path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="synthetic_reviews.csv")
    parser.add_argument("--n-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_synthetic_csv(args.out, n_rows=args.n_rows, seed=args.seed)
    print(f"Wrote {args.n_rows} synthetic rows to {args.out}")
