from pathlib import Path
import requests

URL="https://aact.ctti.clinicaltrials.org/static/exported_files/20260823_export_ctgov.zip"
OUT=Path("data/raw/20260823_export_ctgov.zip")

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    print("Downloading AACT (~2.3 GB). This may take time.")
    with requests.get(URL,stream=True,timeout=120) as r:
        r.raise_for_status()
        with open(OUT,"wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    print(f"Saved to {OUT}")

if __name__=="__main__": main()
