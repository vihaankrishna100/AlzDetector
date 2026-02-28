"""
Run CrewAI agents on first 10 subjects for comprehensive multi-agent analysis.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))

import os
import json
from dotenv import load_dotenv

load_dotenv('/Users/vihaankrishna/ADNI_PROJECT/.env')

from agents.crew_runner import run_adgent_for_subject

import pandas as pd

EXPLAIN_OUT = "explain_outputs"
os.makedirs(EXPLAIN_OUT, exist_ok=True)

clinical_df = pd.read_csv('clinical_100_subjects.csv')
subjects = clinical_df["subject_id"].tolist()[:10]  

print("="*70)
print("MULTI-AGENT INFERENCE (CrewAI) - First 10 Subjects")
print("="*70)

all_results = []

for i, subject_id in enumerate(subjects, 1):
    print(f"\n[{i}/10] Processing {subject_id}...")
    try:
        result = run_adgent_for_subject(subject_id)
        all_results.append({
            "subject_id": subject_id,
            "status": "success",
            "result": result
        })
        print(f"Completed: {subject_id}")
        
        if isinstance(result, dict) and 'output' in result:
            print(f"   Output: {str(result['output'])[:200]}...")
    except Exception as e:
        print(f"Error for {subject_id}: {str(e)[:100]}")
        all_results.append({
            "subject_id": subject_id,
            "status": "failed",
            "error": str(e)
        })

output_file = os.path.join(EXPLAIN_OUT, "agents_batch_10subjects.json")
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n" + "="*70)
print(f"Agent inference complete!")
print(f"Results saved to: {output_file}")
print("="*70)

successful = sum(1 for r in all_results if r["status"] == "success")
print(f"\nSummary: {successful}/10 subjects processed successfully")
