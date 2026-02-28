"""
Comprehensive evaluation metrics for the fused model on all 100 subjects.
"""
import sys
sys.path.insert(0, '/Users/vihaankrishna/ADNI_PROJECT')

import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from adgent.efficient_netv2 import EffNetV2Clinical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "model.pth"
CLINICAL_CSV = "data/clinical_100_subjects.csv"
MRI_ROOT = "NIFTI_ONE_PER_SUBJECT"
LABELS_CSV = "labels.csv"
LABELS_CSV = "data/labels.csv"
EXPLAIN_OUT = "explain_outputs"
os.makedirs(EXPLAIN_OUT, exist_ok=True)

print(f"Using device: {DEVICE}")

print("Loading model...")
model = EffNetV2Clinical(num_classes=2, clin_dim=3).to(DEVICE)
state = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

clinical_df = pd.read_csv(CLINICAL_CSV)

print("Loading ground truth labels...")
y_true = clinical_df["label"].values  
y_pred_proba = []
y_pred = []

print("\nRunning inference on all 100 subjects...")
for subject_id in clinical_df["subject_id"]:
    try:
        row = clinical_df[clinical_df["subject_id"] == subject_id].iloc[0]
        clin_feats = torch.tensor([
            float(row["entry_age"]),
            float(row["total13"]),
            float(row["apoe4_count"])
        ], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        candidates = [f for f in os.listdir(MRI_ROOT) if subject_id in f and f.endswith(".nii.gz")]
        mri_path = os.path.join(MRI_ROOT, candidates[0])
        
        nii = nib.load(mri_path).get_fdata()
        z = nii.shape[2] // 2
        img = nii[:, :, z]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.repeat(1, 3, 1, 1).to(DEVICE)
        
        with torch.no_grad():
            logits, _, _ = model(img_tensor, clin_feats)
            probs = F.softmax(logits, dim=1)
            p_ad = float(probs[0, 1].item())
        
        y_pred_proba.append(p_ad)
        y_pred.append(1 if p_ad > 0.5 else 0)
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        y_pred_proba.append(0.5)
        y_pred.append(0)

y_pred_proba = np.array(y_pred_proba)
y_pred = np.array(y_pred)

print("\n" + "="*70)
print("COMPREHENSIVE EVALUATION METRICS")
print("="*70)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n CLASSIFICATION METRICS (Binary Threshold: 0.5)")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f} (TP / (TP + FP))")
print(f"  Recall:    {recall:.4f} (TP / (TP + FN)) - Sensitivity")
print(f"  F1 Score:  {f1:.4f} (Harmonic mean of precision & recall)")

auroc = roc_auc_score(y_true, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

print(f"\n ROC-AUC METRICS")
print(f"  AUROC: {auroc:.4f} (Area under ROC curve)")
print(f"  Interpretation: {auroc*100:.1f}% probability model ranks random AD higher than random CN")

ap = average_precision_score(y_true, y_pred_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)

print(f"\n PRECISION-RECALL METRICS")
print(f"  AP (Avg Precision): {ap:.4f}")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n CONFUSION MATRIX")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall  
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  
ppv = precision  

print(f"\n ADDITIONAL METRICS")
print(f"  Sensitivity (TPR):    {sensitivity:.4f} - True positive rate")
print(f"  Specificity (TNR):    {specificity:.4f} - True negative rate")
print(f"  PPV (Precision):      {ppv:.4f} - Positive predictive value")
print(f"  NPV:                  {npv:.4f} - Negative predictive value")

print(f"\n PER-CLASS BREAKDOWN")
print(f"\nClass 0 (CN - Cognitively Normal):")
print(f"  Support: {(y_true == 0).sum()}")
cn_precision = precision_score(y_true, y_pred, labels=[0], zero_division=0)
cn_recall = recall_score(y_true, y_pred, labels=[0], zero_division=0)
cn_f1 = f1_score(y_true, y_pred, labels=[0], zero_division=0)
print(f"  Precision: {cn_precision:.4f}")
print(f"  Recall: {cn_recall:.4f}")
print(f"  F1: {cn_f1:.4f}")

print(f"\nClass 1 (AD - Alzheimer's Disease):")
print(f"  Support: {(y_true == 1).sum()}")
ad_precision = precision_score(y_true, y_pred, labels=[1], zero_division=0)
ad_recall = recall_score(y_true, y_pred, labels=[1], zero_division=0)
ad_f1 = f1_score(y_true, y_pred, labels=[1], zero_division=0)
print(f"  Precision: {ad_precision:.4f}")
print(f"  Recall: {ad_recall:.4f}")
print(f"  F1: {ad_f1:.4f}")

print(f"\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, target_names=["CN", "AD"]))

print(f"\n" + "="*70)
print("THRESHOLD ANALYSIS - Optimal Operating Points")
print("="*70)

best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.3, 0.8, 0.05):
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1_thresh = f1_score(y_true, y_pred_thresh)
    if f1_thresh > best_f1:
        best_f1 = f1_thresh
        best_threshold = threshold

y_pred_best = (y_pred_proba >= best_threshold).astype(int)
recall_best = recall_score(y_true, y_pred_best)
precision_best = precision_score(y_true, y_pred_best)

print(f"\nOptimal F1 Threshold: {best_threshold:.3f}")
print(f"  F1 Score: {best_f1:.4f}")
print(f"  Recall: {recall_best:.4f}")
print(f"  Precision: {precision_best:.4f}")

metrics_dict = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "AUROC": auroc,
    "Avg Precision": ap,
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "Threshold": 0.5
}

summary_df = pd.DataFrame([metrics_dict]).T
summary_df.columns = ["Value"]
summary_df["Value"] = summary_df["Value"].apply(lambda x: f"{x:.4f}")

print(f"\n" + "="*70)
print("METRICS SUMMARY TABLE")
print("="*70)
print(summary_df)

metrics_output = {
    "model": "EffNetV2Clinical (MRI + Clinical Fusion)",
    "dataset": "ADNI 100 subjects (baseline)",
    "test_size": len(y_true),
    "class_distribution": {
        "CN": int((y_true == 0).sum()),
        "AD": int((y_true == 1).sum())
    },
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auroc": float(auroc),
        "average_precision": float(ap),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "ppv": float(ppv)
    },
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "per_class_metrics": {
        "CN": {
            "support": int((y_true == 0).sum()),
            "precision": float(cn_precision),
            "recall": float(cn_recall),
            "f1": float(cn_f1)
        },
        "AD": {
            "support": int((y_true == 1).sum()),
            "precision": float(ad_precision),
            "recall": float(ad_recall),
            "f1": float(ad_f1)
        }
    }
}

metrics_file = os.path.join(EXPLAIN_OUT, "evaluation_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(metrics_output, f, indent=2)

print(f"\n✅ Metrics saved to: {metrics_file}")

results_df = clinical_df.copy()
results_df["p_AD"] = y_pred_proba
results_df["pred_label"] = y_pred
results_df["pred_label_str"] = results_df["pred_label"].map({0: "CN", 1: "AD"})
results_df["correct"] = (results_df["label"] == y_pred).astype(int)

results_csv = os.path.join(EXPLAIN_OUT, "detailed_predictions.csv")
results_df.to_csv(results_csv, index=False)
print(f"✅ Detailed predictions saved to: {results_csv}")

print(f"\n" + "="*70)
print("✅ EVALUATION COMPLETE")
print("="*70)
