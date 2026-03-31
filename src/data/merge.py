"""
Data merging utilities.

This module handles combining multiple season datasets
into a single unified dataset.
"""

import pandas as pd


def merge_seasons(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple season DataFrames into one dataset.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        List of season DataFrames.

    Returns
    -------
    pd.DataFrame
        Combined dataset.
    """
    if len(dataframes) == 0:
        raise ValueError("No dataframes provided for merging.")

    merged_df = pd.concat(dataframes, ignore_index=True)

    return merged_df