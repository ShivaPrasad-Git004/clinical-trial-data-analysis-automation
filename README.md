# Clinical Trial Data Analysis and Automation

A complete starter project for **Automated Clinical Trial Data Quality Analysis and Anomaly Detection Using Python, SQL and Machine Learning**.

## Included
- 5,025-row synthetic starter dataset with intentional data-quality problems.
- AACT downloader for the real public clinical-trial metadata.
- ETL/cleaning, validation, quality scoring and Isolation Forest anomaly detection.
- MySQL schema + analytical SQL queries.
- Streamlit dashboard.
- Power BI-ready CSV.
- Automated Excel/CSV quality reports.
- `.env.example` for MySQL connection.

## Important
The included CSV is **synthetic demo data**, not real patient data. It lets you run the project immediately.
For the research version, download AACT data using `python -m src.download_aact`. The current AACT snapshots are about 2.3 GB, so the real database is not bundled.

AACT: https://aact.ctti.clinicaltrials.org/

## Run
1. `python -m venv .venv`
2. Windows: `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python -m src.pipeline`
5. `streamlit run dashboard/app.py`

## MySQL
1. Start MySQL 8.x.
2. Run `sql/schema.sql` in MySQL Workbench.
3. Copy `.env.example` to `.env` and set your password.
4. Run `python -m src.load_mysql`
5. Run queries in `sql/analysis_queries.sql`

## Real AACT workflow
1. `python -m src.download_aact`
2. Extract the ZIP to `data/raw/aact/`
3. Ensure `studies.txt`, `conditions.txt`, `interventions.txt`, and `sponsors.txt` exist.
4. `python -m src.aact_prepare`
5. Adapt the pipeline input to `data/processed/aact_master.csv`.

AACT tables are linked primarily by `nct_id`. The one-to-many tables are aggregated before joining to avoid duplicate row multiplication.
