
"""Load the processed clinical-trial dataset into MySQL."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from .parsing import read_csv_safe

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "clinical_trials_master.csv"
TABLE_NAME = "clinical_trials"

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "clinical_trial_db")


def load_to_mysql():
    """Load the complete processed CSV into the MySQL table."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {CSV_PATH}\n"
            "Run the data pipeline first."
        )

    # Preserve literal values such as phase='NA'.
    df = read_csv_safe(CSV_PATH)

    if df.empty:
        raise ValueError(
            f"The processed dataset is empty: {CSV_PATH}"
        )

    # URL.create safely handles special characters in passwords.
    database_url = URL.create(
        drivername="mysql+pymysql",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

    engine = create_engine(database_url, pool_pre_ping=True)

    try:
        # WARNING: replace drops and recreates the existing table.
        df.to_sql(
            TABLE_NAME,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=1000,
        )

        print(f"Loaded {len(df):,} records into MySQL.")
        print(f"Table: {DB_NAME}.{TABLE_NAME}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    load_to_mysql()
