from pathlib import Path
import pandas as pd

def load_data(file_path):
    """
    Load dataset from a file path.
    Currently supports CSV, but structured to be easily extended.
    """

    file_path = Path(file_path)

    # If no extension provided, assume CSV
    if file_path.suffix == "":
        file_path = file_path.with_suffix(".csv")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    elif file_path.suffix.lower() == ".xlsx":
        return pd.read_excel(file_path)

    elif file_path.suffix.lower() == ".json":
        return pd.read_json(file_path)

    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def export_data(df, file_path, index=False):
    """
    Export a pandas DataFrame to multiple file formats.

    Supported formats:
    - .csv
    - .xlsx
    - .json
    - .parquet
    """

    file_path = Path(file_path)

    # If no extension provided, default to CSV
    if file_path.suffix == "":
        file_path = file_path.with_suffix(".csv")

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(file_path, index=index)

    elif suffix == ".xlsx":
        df.to_excel(file_path, index=index)

    elif suffix == ".json":
        df.to_json(file_path, orient="records", indent=4)

    elif suffix == ".parquet":
        df.to_parquet(file_path, index=index)

    else:
        raise ValueError(f"Unsupported export format: {suffix}")

    return str(file_path)