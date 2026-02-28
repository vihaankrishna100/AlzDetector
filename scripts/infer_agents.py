from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.crew_runner import run_adgent_for_subject
import pandas as pd

df = pd.read_csv("clinical_100_subjects.csv")

for i in range(min(20, len(df))):
    sid = df.iloc[i]["subject_id"]
    print("\n==========================")
    print("Running ADgent on:", sid)
    out = run_adgent_for_subject(sid)
    print(out["raw_output"])

