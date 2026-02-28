#!/usr/bin/env python3

import sys
import os
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.crew_runner import run_adgent_for_subject

def extract_final_label(agent_output: str) -> int:
    """
    Extract final_label (0 or 1) from agent output JSON.
    The supervisor task returns structured JSON with 'final_label' key.
    JSON may be embedded in markdown code blocks.
    """
    # First try: extract JSON from markdown code block ```json ... ```
    try:
        json_match = re.search(r'```json\s*\n(.*?)\n```', agent_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            label = data.get("final_label")
            if isinstance(label, str):
                return 1 if label.upper() == "AD" else 0
            elif isinstance(label, int):
                return int(label)
    except Exception:
        pass
    
    # Second try: search for JSON object with final_label key
    try:
        json_match = re.search(r'\{[^{}]*"final_label"\s*:\s*"([^"]+)"', agent_output, re.DOTALL)
        if json_match:
            label_str = json_match.group(1).upper()
            return 1 if label_str == "AD" else 0
    except Exception:
        pass
    
    # Fallback: search for "final_label" with any pattern
    try:
        if "final_label" in agent_output:
            if "AD" in agent_output and agent_output.index("AD") < agent_output.index("final_label") + 100:
                return 1
            elif "CN" in agent_output and agent_output.index("CN") < agent_output.index("final_label") + 100:
                return 0
    except Exception:
        pass
    
    # Return None if unparseable
    return None


def evaluate_multiagent(test_only: bool = False, test_count: int = 5):
    """Run multi-agent on all subjects, compare to ground truth, compute metrics."""
    
    # Load ground truth labels
    labels_path = PROJECT_ROOT / "data/labels.csv"
    if not labels_path.exists():
        print(f"Error: {labels_path} not found")
        return
    
    labels_df = pd.read_csv(labels_path)
    
    if test_only:
        labels_df = labels_df.head(test_count)
        print(f"[TEST MODE] Running on {len(labels_df)} subjects...\n")
    else:
        print(f"Loaded {len(labels_df)} ground truth labels from {labels_path.name}")
        print(f"Columns: {labels_df.columns.tolist()}")
        print(f"Class distribution: {labels_df['label'].value_counts().to_dict()}\n")
    
    #run multi-agent inference on each subject
    predictions = []
    ground_truth = []
    subject_results = []
    
    for idx, row in labels_df.iterrows():
        subject_id = row["subject_id"]

        true_label = int(row["label"])  #0=CN, 1=AD
        
        print(f"[{idx + 1}/{len(labels_df)}] Running agents for {subject_id}...", end=" ", flush=True)
        #error handling after 
        try:
            #nkRun the multi-agent crew
            agent_result = run_adgent_for_subject(subject_id)
            raw_output = agent_result.get("raw_output", "")
            
            # Extract predicted label
            pred_label = extract_final_label(raw_output)
            
            if pred_label is None:
                print("ERROR: Could not parse output")

                subject_results.append({
                    "subject_id": subject_id,
                    "ground_truth": true_label,
                    "predicted": None,
                    "correct": False,
                })
                continue
            
            predictions.append(pred_label)
            ground_truth.append(true_label)
            
            pred_str = "AD" if pred_label == 1 else "CN"
            true_str = "AD" if true_label == 1 else "CN"
            match = "✓" if pred_label == true_label else "✗"
            
            print(f"Pred={pred_str} (True={true_str}) {match}")
            
            subject_results.append({
                "subject_id": subject_id,
                "ground_truth": true_label,
                "predicted": pred_label,
                "correct": pred_label == true_label,
            })
        
        except Exception as e:
            print(f"ERROR: {e}")
            subject_results.append({
                "subject_id": subject_id,
                "ground_truth": true_label,
                "predicted": None,

                "correct": False,
            })
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("MULTI-AGENT EVALUATION RESULTS")
    print("=" * 70 + "\n")
    
    # Filter out None predictions for metrics
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    valid_preds = [predictions[i] for i in valid_indices]
    valid_truths = [ground_truth[i] for i in valid_indices]
    
    if not valid_preds:
        print("Error: No valid predictions generated.")
        return
    
    acc = accuracy_score(valid_truths, valid_preds)
    prec = precision_score(valid_truths, valid_preds, zero_division=0)
    rec = recall_score(valid_truths, valid_preds, zero_division=0)
    f1 = f1_score(valid_truths, valid_preds, zero_division=0)
    cm = confusion_matrix(valid_truths, valid_preds)
    
    print(f"Total subjects evaluated   : {len(labels_df)}")
    print(f"Valid predictions           : {len(valid_preds)}")
    print(f"\nAccuracy                   : {acc:.4f}")
    print(f"Precision (AD)             : {prec:.4f}")
    print(f"Recall (AD)                : {rec:.4f}")
    print(f"F1-score (AD)              : {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN (CN→CN) : {cm[0, 0]}")
    print(f"  FP (CN→AD) : {cm[0, 1]}")
    print(f"  FN (AD→CN) : {cm[1, 0]}")
    print(f"  TP (AD→AD) : {cm[1, 1]}")
    
    print(f"\nClassification Report:")
    print(classification_report(valid_truths, valid_preds, target_names=["CN (0)", "AD (1)"], zero_division=0))
    
    # Save results
    results_df = pd.DataFrame(subject_results)
    output_csv = PROJECT_ROOT / "multiagent_evaluation_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")
    
    # Summary metrics
    summary = {
        "total_subjects": len(labels_df),
        "valid_predictions": len(valid_preds),
        "accuracy": float(acc),
        "precision_ad": float(prec),
        "recall_ad": float(rec),
        "f1_ad": float(f1),
        "confusion_matrix": cm.tolist(),
    }
    
    output_json = PROJECT_ROOT / "multiagent_evaluation_summary.json"
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics saved to: {output_json}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multi-agent accuracy on ADNI subjects")
    parser.add_argument("--test", action="store_true", help="Run in test mode on first 5 subjects")
    parser.add_argument("--test-count", type=int, default=5, help="Number of subjects in test mode")
    args = parser.parse_args()
    
    evaluate_multiagent(test_only=args.test, test_count=args.test_count)
