"""Random Forest baseline, built as an sklearn Pipeline.

Replaces RF.ipynb's manual `scipy.sparse.hstack` of a CountVectorizer matrix with the
numeric feature columns -- a ColumnTransformer does the same thing declaratively, and the
resulting Pipeline can be fit/predicted on a plain DataFrame.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Best hyperparameters from the thesis's random search (Section 5.1.1 / Appendix A, Table 2).
DEFAULT_RF_PARAMS = dict(
    n_estimators=1000,
    bootstrap=True,
    max_samples=0.5,
    max_features="sqrt",
    oob_score=True,
    min_samples_leaf=1,
)


def build_rf_pipeline(
    extra_feature_columns: Sequence[str] = (),
    class_weight: Optional[str] = None,
    text_column: str = "combined_text",
    random_state: int = 42,
    **rf_overrides,
) -> Pipeline:
    transformers = [("text", CountVectorizer(max_features=1000, stop_words="english"), text_column)]
    if extra_feature_columns:
        transformers.append(("extra", "passthrough", list(extra_feature_columns)))
    preprocessor = ColumnTransformer(transformers)

    rf_params = {**DEFAULT_RF_PARAMS, **rf_overrides}
    rf = RandomForestClassifier(random_state=random_state, class_weight=class_weight, **rf_params)
    return Pipeline([("features", preprocessor), ("rf", rf)])
