"""
Data loading utilities.

This module is responsible for safely loading raw data files
(CSV and Excel) into pandas DataFrames.

"""

from pathlib import Path
import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def load_excel(file_path: Path, sheet_name=0) -> pd.DataFrame:
    """
    Load a spreadsheet file into a pandas DataFrame.

    Supports:
    - .xlsx via openpyxl
    - .ods via odf

    Parameters
    ----------
    file_path : Path
        Path to the spreadsheet file.
    sheet_name : str or int, optional
        Sheet to read (default is first sheet).

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

    if suffix == ".ods":
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="odf")

    raise ValueError(
        f"Unsupported spreadsheet format: {suffix}. "
        "Supported formats are .xlsx and .ods"
    )