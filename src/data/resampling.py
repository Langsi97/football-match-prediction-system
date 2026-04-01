"""
Training-only resampling utilities.

This module applies SMOTE to the training set only.
It must be used after preprocessing and never on the test set.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str | dict = "not majority",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training data only.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training target labels.
    sampling_strategy : str | dict, default="not majority"
        SMOTE sampling strategy.
        - "not majority": oversample all minority classes up to majority class size
        - dict: explicit class target counts
    random_state : int, default=42
        Random seed for reproducibility.
    k_neighbors : int, default=5
        Number of nearest neighbors used by SMOTE.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        SMOTE-resampled X_train and y_train.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name if y_train.name else "FTR")

    return X_resampled, y_resampled