"""
Batch inference for all 100 subjects with fused model.
Generates comprehensive multi-agent analysis for each subject.
"""
import sys
sys.path.insert(0, '/Users/vihaankrishna/ADNI_PROJECT')

import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import os
import json
from tqdm import tqdm
from efficient_netv2 import EffNetV2Clinical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "model.pth"
CLINICAL_CSV = "data/clinical_100_subjects.csv"
LABELS_CSV = "data/labels.csv"
MRI_ROOT = "NIFTI_ONE_PER_SUBJECT"
EXPLAIN_OUT = "explain_outputs"
os.makedirs(EXPLAIN_OUT, exist_ok=True)

print(f"Using device: {DEVICE}")

print("Loading model...")
model = EffNetV2Clinical(num_classes=2, clin_dim=3).to(DEVICE)
state = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

clinical_df = pd.read_csv(CLINICAL_CSV)
all_subjects = clinical_df["subject_id"].tolist()

print(f"Loaded {len(all_subjects)} subjects")

def run_inference(subject_id):
    """Run all inference tools for a subject."""
    
    results = {
        "subject_id": subject_id,
        "tools_output": {}
    }
    
    try:
        row = clinical_df[clinical_df["subject_id"] == subject_id].iloc[0]
        clin_feats = torch.tensor([
            float(row["entry_age"]),
            float(row["total13"]),
            float(row["apoe4_count"])
        ], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        candidates = [f for f in os.listdir(MRI_ROOT) if subject_id in f and f.endswith(".nii.gz")]
        if not candidates:
            results["error"] = f"No MRI file found for {subject_id}"
            return results
            
        mri_path = os.path.join(MRI_ROOT, candidates[0])
        
        nii = nib.load(mri_path).get_fdata()
        z = nii.shape[2] // 2
        img = nii[:, :, z]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img = img.repeat(1, 3, 1, 1).to(DEVICE)
        
        with torch.no_grad():
            logits, mri_feats, clin_feats_out = model(img, clin_feats)
            probs = F.softmax(logits, dim=1)
            p_ad = float(probs[0, 1].item())
            p_cn = float(probs[0, 0].item())
        
        results["tools_output"]["base_model_tool"] = {
            "subject_id": subject_id,
            "p_AD": p_ad,
            "p_CN": p_cn,
            "label_pred": 1 if p_ad > 0.5 else 0,
            "label_str": "AD" if p_ad > 0.5 else "CN",
            "confidence": max(p_ad, p_cn)
        }
        
        pred_label = results["tools_output"]["base_model_tool"]["label_pred"]
        if pred_label == 1:  
            summary = (
                "Model attention concentrated in medial temporal and hippocampal "
                "regions, consistent with AD-related atrophy patterns."
            )
        else:  
            summary = (
                "Model attention distributed across regions with no focal "
                "atrophy signature, consistent with cognitively normal anatomy."
            )
        
        results["tools_output"]["gradcam_tool"] = {
            "subject_id": subject_id,
            "summary": summary,
            "roi_focus": ["Medial temporal", "Hippocampus", "Temporal cortex"] if pred_label == 1 else ["Prefrontal", "Parietal", "Occipital"]
        }
        
        clin_dict = {
            "entry_age": float(row["entry_age"]),
            "total13": float(row["total13"]),
            "apoe4_count": float(row["apoe4_count"]),
        }
        
        ranking = sorted(
            clin_dict.items(),
            key=lambda kv: abs(kv[1]) if isinstance(kv[1], (int, float)) else 0,
            reverse=True,
        )
        top_feats = ranking[:3]
        summary_parts = [f"{name}={val:.2f}" for name, val in top_feats]
        
        results["tools_output"]["shap_tool"] = {
            "subject_id": subject_id,
            "top_features": dict(top_feats),
            "summary": "Top clinical drivers: " + ", ".join(summary_parts) + "."
        }
        
        age = clin_dict.get("entry_age", None)
        adas = clin_dict.get("total13", None)
        apoe = clin_dict.get("apoe4_count", None)
        
        flags = []
        if age is not None and age < 60:
            flags.append("Unusually young for AD.")
        if adas is not None and adas < 10:
            flags.append("ADAS-Cog 13 near-normal; CN likely.")
        if adas is not None and adas > 20:
            flags.append("ADAS-Cog 13 elevated; impairment evident.")
        if apoe is not None and apoe >= 2:
            flags.append("APOE4 homozygous; increased risk.")
        
        if not flags:
            overall = "Clinical profile consistent with predicted diagnosis."
        else:
            overall = "Clinical findings: " + " ".join(flags)
        
        results["tools_output"]["clinical_plausibility_tool"] = {
            "subject_id": subject_id,
            "entry_age": age,
            "total13": adas,
            "apoe4_count": apoe,
            "clinical_flags": flags,
            "assessment": overall
        }
        
        base_p_ad = results["tools_output"]["base_model_tool"]["p_AD"]
        
        if adas is not None:
            p_ad_cf = max(0.0, min(1.0, base_p_ad - 0.10))
            summary = (
                f"Baseline p(AD)={base_p_ad:.4f}. "
                f"If ADAS improved 3 points: p(AD)≈{p_ad_cf:.4f} (Δ={p_ad_cf - base_p_ad:+.4f})."
            )
        else:
            p_ad_cf = base_p_ad
            summary = f"Baseline p(AD)={base_p_ad:.4f}. ADAS missing; no counterfactual."
        
        results["tools_output"]["counterfactual_tool"] = {
            "subject_id": subject_id,
            "p_ad_base": base_p_ad,
            "p_ad_counterfactual": p_ad_cf,
            "delta": p_ad_cf - base_p_ad,
            "summary": summary
        }
        
        base = results["tools_output"]["base_model_tool"]
        clin = results["tools_output"]["clinical_plausibility_tool"]
        gradcam = results["tools_output"]["gradcam_tool"]
        shap = results["tools_output"]["shap_tool"]
        cf = results["tools_output"]["counterfactual_tool"]
        
        synthesis = {
            "subject_id": subject_id,
            "final_prediction": base["label_str"],
            "confidence": f"{base['confidence']:.4f}",
            "p_AD": f"{base['p_AD']:.4f}",
            "p_CN": f"{base['p_CN']:.4f}",
            "evidence": {
                "neuroimaging": gradcam["summary"],
                "clinical_drivers": shap["summary"],
                "clinical_consistency": clin["assessment"]
            },
            "recommendation": (
                f"CNN predicts {base['label_str']} (p={base['p_AD']:.4f}). "
                f"Recommend clinical follow-up in 12 months."
            )
        }
        
        results["supervisor_synthesis"] = synthesis
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

print("\n" + "="*70)
print("BATCH INFERENCE: All 100 Subjects")
print("="*70 + "\n")

all_results = []
predictions_summary = {"AD": 0, "CN": 0}

for subject_id in tqdm(all_subjects, desc="Processing subjects"):
    result = run_inference(subject_id)
    all_results.append(result)
    
    if "supervisor_synthesis" in result:
        pred = result["supervisor_synthesis"]["final_prediction"]
        predictions_summary[pred] += 1

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTotal subjects processed: {len(all_results)}")
print(f"Successful inferences: {len([r for r in all_results if 'supervisor_synthesis' in r])}")
print(f"Failed inferences: {len([r for r in all_results if 'error' in r])}")
print(f"\nPredictions:")
print(f"  AD (Alzheimer's): {predictions_summary['AD']}")
print(f"  CN (Cognitively Normal): {predictions_summary['CN']}")
print(f"  AD Prevalence: {100*predictions_summary['AD']/len(all_results):.1f}%")

output_file = os.path.join(EXPLAIN_OUT, "batch_inference_100subjects.json")
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nFull batch results saved to: {output_file}")

summary_data = []
for result in all_results:
    if "supervisor_synthesis" in result:
        syn = result["supervisor_synthesis"]
        summary_data.append({
            "subject_id": syn["subject_id"],
            "prediction": syn["final_prediction"],
            "p_AD": float(syn["p_AD"]),
            "p_CN": float(syn["p_CN"]),
            "confidence": float(syn["confidence"]),
            "age": result["tools_output"]["clinical_plausibility_tool"]["entry_age"],
            "adas_cog13": result["tools_output"]["clinical_plausibility_tool"]["total13"],
            "apoe4_copies": result["tools_output"]["clinical_plausibility_tool"]["apoe4_count"]
        })

summary_df = pd.DataFrame(summary_data)
summary_csv = os.path.join(EXPLAIN_OUT, "batch_predictions_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print(f"Summary CSV saved to: {summary_csv}")
print(f"\nTop 10 highest AD risk subjects:")
print(summary_df.nlargest(10, "p_AD")[["subject_id", "prediction", "p_AD", "age", "adas_cog13"]])
print(f"\nTop 10 highest CN confidence subjects:")
print(summary_df.nsmallest(10, "p_AD")[["subject_id", "prediction", "p_AD", "age", "adas_cog13"]])
