"""
Training-only resampling utilities.

This module applies simple class oversampling to the training set only.
It is intended to reproduce notebook behavior in a production-grade way
without touching the test set.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def oversample_target_classes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_classes: list[str] | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversample selected target classes up to the majority class size.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    target_classes : list[str] | None
        Classes to oversample. If None, all minority classes are oversampled.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Resampled training features and target.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    working_df = X_train.copy()
    working_df["FTR"] = y_train.values

    class_counts = working_df["FTR"].value_counts()
    max_count = class_counts.max()

    if target_classes is None:
        target_classes = class_counts[class_counts < max_count].index.tolist()

    resampled_parts = []

    for class_label, group_df in working_df.groupby("FTR"):
        if class_label in target_classes:
            n_to_add = max_count - len(group_df)
            if n_to_add > 0:
                sampled_df = group_df.sample(
                    n=n_to_add,
                    replace=True,
                    random_state=random_state,
                )
                group_df = pd.concat([group_df, sampled_df], axis=0)

        resampled_parts.append(group_df)

    resampled_df = (
        pd.concat(resampled_parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    X_resampled = resampled_df.drop(columns=["FTR"])
    y_resampled = resampled_df["FTR"]

    return X_resampled, y_resampled