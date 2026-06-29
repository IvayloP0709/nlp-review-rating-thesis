"""Loading, splitting, and balancing for the review-rating dataset.

The original Google Local Data merge and feature engineering (Section 4.3 of the thesis:
joining reviews to businesses on `gmap_id`, spaCy NER for `general_personal`, price-tier
parsing, response timing/length) happened in a preprocessing stage that was never checked
into this repository -- every script here always started from an already-preprocessed CSV.
This module reflects that: it expects data already in that shape (see EXTRA_FEATURE_COLUMNS)
and covers the parts that *were* implemented and duplicated across every run script: label
encoding, splitting, balancing, and combining the two text fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

RATING_TO_LABEL = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
LABEL_TO_RATING = {v: k for k, v in RATING_TO_LABEL.items()}
NUM_CLASSES = len(RATING_TO_LABEL)
EXTRA_FEATURE_COLUMNS = ["response_timing", "response_length", "general_personal", "price_range"]


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_preprocessed(csv_path: str) -> pd.DataFrame:
    """Load the already-cleaned dataset and encode `rating` (1.0-5.0) as an ordinal `label` (0-4)."""
    df = pd.read_csv(csv_path)
    df["label"] = df["rating"].map(RATING_TO_LABEL)
    return df


def make_splits(df: pd.DataFrame, random_state: int = 42) -> Splits:
    """Stratified 60/20/20 train/val/test split, matching every original run script."""
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val["label"], random_state=random_state
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """Concatenate review + managerial-response text into one string per row.

    The original scripts built this with `str(value)`, which turns a missing `resp_text`
    (~89% of rows -- see RF.ipynb's df.info()) into the literal substring "nan" appended to
    the review. Use fillna("") instead so missing responses contribute nothing.
    """
    review = df["text"].fillna("").astype(str)
    response = df["resp_text"].fillna("").astype(str)
    return (review + " " + response).str.strip()


def oversample(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Random oversampling of the minority classes, train split only."""
    return RandomOverSampler(random_state=random_state).fit_resample(X, y)


def class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Per-class weights for `nn.CrossEntropyLoss(weight=...)`, indexed by class id.

    The original `compute_class_weights` returned a per-*sample* tensor (length len(y_train))
    and then sliced its first `num_classes` entries by position, treating "the weight of the
    i-th training row" as if it were "the weight for class i". This computes one weight per
    class instead, in class-id order.
    """
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(weights, dtype=torch.float)
