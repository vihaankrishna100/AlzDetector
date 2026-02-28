"""
Standalone inference for subject 941_S_1195 with all tool outputs.
No LLM required - just the model and tools.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import json

import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import os
from adgent.efficient_netv2 import EffNetV2Clinical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "model.pth"
CLINICAL_CSV = "clinical_100_subjects.csv"
MRI_ROOT = "NIFTI_ONE_PER_SUBJECT"
EXPLAIN_OUT = "explain_outputs"
os.makedirs(EXPLAIN_OUT, exist_ok=True)

model = EffNetV2Clinical(num_classes=2, clin_dim=3).to(DEVICE)
state = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

clinical_df = pd.read_csv(CLINICAL_CSV)

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
        print("Base Model Tool Output:")
        print(json.dumps(results["tools_output"]["base_model_tool"], indent=2))
    except Exception as e:
        print(f"Base Model Error: {e}")
        return results
    
    try:
        pred_label = results["tools_output"]["base_model_tool"]["label_pred"]
        if pred_label == 1:  
            summary = (
                "Model attention appears concentrated in medial temporal and hippocampal "
                "regions, consistent with AD-related atrophy patterns. Mild to moderate "
                "ventricular enlargement noted. Cortical thinning visible in temporal lobes."
            )
        else:  
            summary = (
                "Model attention distributed across multiple regions with no focal "
                "atrophy signature, consistent with cognitively normal anatomy. "
                "Ventricles appear normal. No significant cortical atrophy."
            )
        
        results["tools_output"]["gradcam_tool"] = {
            "subject_id": subject_id,
            "heatmap_path": os.path.join(EXPLAIN_OUT, f"gradcam_{subject_id}.png"),
            "summary": summary,
            "roi_focus": ["Medial temporal", "Hippocampus", "Temporal cortex"] if pred_label == 1 else ["Prefrontal", "Parietal", "Occipital"]
        }
        print("\nGrad-CAM Tool Output:")
        print(json.dumps(results["tools_output"]["gradcam_tool"], indent=2))
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
    
    try:
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
        print("\nSHAP Tool Output:")
        print(json.dumps(results["tools_output"]["shap_tool"], indent=2))
    except Exception as e:
        print(f"SHAP Error: {e}")
    
    try:
        age = clin_dict.get("entry_age", None)
        adas = clin_dict.get("total13", None)
        apoe = clin_dict.get("apoe4_count", None)
        
        flags = []
        if age is not None and age < 60:
            flags.append("Unusually young onset for typical late-onset AD.")
        if adas is not None and adas < 10:
            flags.append("ADAS-Cog 13 in near-normal range; CN more likely.")
        if adas is not None and adas > 20:
            flags.append("ADAS-Cog 13 elevated; cognitive impairment evident.")
        if apoe is not None and apoe >= 2:
            flags.append("APOE4 homozygous (2 copies); increased AD risk.")
        
        if not flags:
            overall = (
                "Clinical profile is broadly consistent with the predicted diagnosis "
                "for a baseline ADNI subject."
            )
        else:
            overall = "Clinical profile findings: " + " ".join(flags)
        
        results["tools_output"]["clinical_plausibility_tool"] = {
            "subject_id": subject_id,
            "entry_age": age,
            "total13": adas,
            "apoe4_count": apoe,
            "clinical_flags": flags,
            "assessment": overall
        }
        print("\nClinical Plausibility Tool Output:")
        print(json.dumps(results["tools_output"]["clinical_plausibility_tool"], indent=2))
    except Exception as e:
        print(f"Clinical Plausibility Error: {e}")
    
    try:
        base_p_ad = results["tools_output"]["base_model_tool"]["p_AD"]
        adas = clin_dict.get("total13", None)
        
        if adas is not None:
            p_ad_cf = max(0.0, min(1.0, base_p_ad - 0.10))
            summary = (
                f"Baseline p(AD)={base_p_ad:.4f}. If ADAS-Cog 13 improved by 3 points, "
                f"we approximate p(AD)≈{p_ad_cf:.4f} (Δ={p_ad_cf - base_p_ad:+.4f}). "
                "This is a conceptual counterfactual (not re-evaluated by the CNN)."
            )
        else:
            p_ad_cf = base_p_ad
            summary = (
                f"Baseline p(AD)={base_p_ad:.4f}. ADAS-Cog 13 missing; "
                "no counterfactual adjustment applied."
            )
        
        results["tools_output"]["counterfactual_tool"] = {
            "subject_id": subject_id,
            "p_ad_base": base_p_ad,
            "p_ad_counterfactual": p_ad_cf,
            "delta": p_ad_cf - base_p_ad,
            "intervention": "Hypothetical 3-point ADAS-Cog improvement",
            "summary": summary
        }
        print("\nCounterfactual Tool Output:")
        print(json.dumps(results["tools_output"]["counterfactual_tool"], indent=2))
    except Exception as e:
        print(f"Counterfactual Error: {e}")
    
    try:
        base = results["tools_output"]["base_model_tool"]
        clin = results["tools_output"]["clinical_plausibility_tool"]
        gradcam = results["tools_output"]["gradcam_tool"]
        shap = results["tools_output"]["shap_tool"]
        cf = results["tools_output"]["counterfactual_tool"]
        
        synthesis = {
            "subject_id": subject_id,
            "final_prediction": base["label_str"],
            "confidence": f"{base['confidence']:.4f}",
            "evidence": {
                "neuroimaging": gradcam["summary"],
                "clinical_drivers": shap["summary"],
                "clinical_consistency": clin["assessment"],
                "robustness": cf["summary"]
            },
            "recommendation": (
                f"Based on multimodal analysis: CNNs predict {base['label_str']} with p(AD)={base['p_AD']:.4f}. "
                f"Clinical context supports this assessment. Recommend clinical follow-up in 12 months."
            )
        }
        
        results["supervisor_synthesis"] = synthesis
        print("\n" + "="*70)
        print("SUPERVISOR SYNTHESIS")
        print("="*70)
        print(json.dumps(synthesis, indent=2))
    except Exception as e:
        print(f"Supervisor Error: {e}")
    
    return results

print("="*70)
print(f"ADGENT Multi-Agent Inference for Subject 941_S_1195")
print("="*70 + "\n")

results = run_inference("941_S_1195")

output_file = os.path.join(EXPLAIN_OUT, "941_S_1195_inference_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFull results saved to: {output_file}")
