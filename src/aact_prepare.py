from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AACT=ROOT/"data/raw/aact"
OUT=ROOT/"data/processed/aact_master.csv"

def read_pipe(name,usecols):
    return pd.read_csv(AACT/name,sep="|",usecols=usecols,low_memory=False)

def main():
    studies=read_pipe("studies.txt",["nct_id","brief_title","overall_status","study_type","phase","enrollment","start_date","completion_date"])
    conditions=read_pipe("conditions.txt",["nct_id","name"])
    interventions=read_pipe("interventions.txt",["nct_id","name","intervention_type"])
    sponsors=read_pipe("sponsors.txt",["nct_id","name","agency_class"])
    conditions=conditions.groupby("nct_id")["name"].agg(lambda x:"; ".join(x.dropna().astype(str).unique())).reset_index(name="condition")
    interventions=interventions.groupby("nct_id").agg(
        intervention=("name",lambda x:"; ".join(x.dropna().astype(str).unique())),
        intervention_type=("intervention_type",lambda x:"; ".join(x.dropna().astype(str).unique()))
    ).reset_index()
    sponsors=sponsors.groupby("nct_id").agg(
        sponsor=("name",lambda x:"; ".join(x.dropna().astype(str).unique())),
        agency_class=("agency_class",lambda x:"; ".join(x.dropna().astype(str).unique()))
    ).reset_index()
    master=studies.merge(conditions,on="nct_id",how="left").merge(interventions,on="nct_id",how="left").merge(sponsors,on="nct_id",how="left")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    master.to_csv(OUT,index=False)
    print(f"Created {OUT} with {len(master):,} trials")

if __name__=="__main__": main()
