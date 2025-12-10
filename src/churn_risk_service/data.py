from pathlib import Path
import pandas as pd
import requests

# Default raw dataset path
DATA_RAW_PATH = Path("data") / "raw" / "telco_churn.csv"

# Public IBM-hosted Telco churn CSV (Apache-2.0 licensed)
TELCO_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def _ensure_raw_data(path: Path = DATA_RAW_PATH) -> Path:
    """
    Ensure the raw Telco churn CSV exists locally.

    If the file is missing, download it from TELCO_SOURCE_URL
    and save it to data/raw/telco_churn.csv.
    """
    path = Path(path)

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(TELCO_SOURCE_URL, timeout=30)
    resp.raise_for_status()

    with path.open("wb") as f:
        f.write(resp.content)

    return path


def load_raw_telco_churn(path: str | Path = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw Telco churn dataset from CSV.

    If the file is not found locally, it is downloaded automatically.

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

    # Auto-download if missing
    path = _ensure_raw_data(path)

    df = pd.read_csv(path)

    # Normalize column names a bit
    df.columns = df.columns.str.strip()

    return df


if __name__ == "__main__":
    df = load_raw_telco_churn()
    print(df.head())
    print(f"\nShape: {df.shape}")
