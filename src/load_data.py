import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Locate the SQL file relative to the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_FILE = PROJECT_ROOT / "sql" / "Project_SQL.sql"

# Load local database settings from .env
env_file = PROJECT_ROOT / ".env"

if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator:
            os.environ.setdefault(key.strip(), value.strip())

# Database connection settings
db_url = URL.create(
    "postgresql+psycopg2",
    username=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME", "BAN5003"),
)

def load_data():
    if not os.getenv("DB_PASSWORD"):
        raise RuntimeError("DB_PASSWORD is not set.")

    # Read Matthew's SQL query from its separate file
    sql_query = SQL_FILE.read_text(encoding="utf-8")

    engine = create_engine(db_url)

    with engine.connect() as connection:
        df = pd.read_sql_query(text(sql_query), connection)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())  

    if len(df) != 85571:
        print("Warning: Row count differs from the Stage 1 baseline.")

    engine.dispose()
    return df

if __name__ == "__main__":
    load_data()
