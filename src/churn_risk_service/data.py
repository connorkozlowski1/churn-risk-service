from pathlib import Path
import pandas as pd


# Default raw dataset path
DATA_RAW_PATH = Path("data") / "raw" / "telco_churn.csv"


def load_raw_telco_churn(path: str | Path = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw Telco churn dataset from CSV.

    Parameters
    ----------
    path : str or Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw churn dataset with standardized column names.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw Telco churn file not found at: {path}")

    df = pd.read_csv(path)

    # Normalize column names a bit
    df.columns = df.columns.str.strip()

    return df


if __name__ == "__main__":
    df = load_raw_telco_churn()
    print(df.head())
    print(f"\nShape: {df.shape}")
