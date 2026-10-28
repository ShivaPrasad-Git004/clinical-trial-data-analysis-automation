from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"

def generate_report(df,issues):
    REPORTS.mkdir(exist_ok=True)
    issues.to_csv(REPORTS/"validation_issues.csv",index=False)
    summary=pd.DataFrame({
        "metric":["Total records","Average quality score","Excellent","Good","Moderate","Poor","Critical","Anomalies"],
        "value":[len(df),round(df.quality_score.mean(),2),
                 int((df.quality_status=="EXCELLENT").sum()),
                 int((df.quality_status=="GOOD").sum()),
                 int((df.quality_status=="MODERATE").sum()),
                 int((df.quality_status=="POOR").sum()),
                 int((df.quality_status=="CRITICAL").sum()),
                 int(df.anomaly_flag.sum())]
    })
    summary.to_csv(REPORTS/"quality_summary.csv",index=False)
    with pd.ExcelWriter(REPORTS/"clinical_trial_quality_report.xlsx") as writer:
        summary.to_excel(writer,sheet_name="Summary",index=False)
        issues.to_excel(writer,sheet_name="Validation_Issues",index=False)
        df.head(1000).to_excel(writer,sheet_name="Sample_Data",index=False)
