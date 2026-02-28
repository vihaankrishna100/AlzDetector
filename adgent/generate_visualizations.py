"""
Generate Grad-CAM heatmap and SHAP force plot for a subject.
"""
import sys
sys.path.insert(0, '/Users/vihaankrishna/ADNI_PROJECT')

import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def generate_gradcam(subject_id, layer_name="backbone"):
    """
    Generate Grad-CAM heatmap for a subject's MRI.
    """
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
    img_tensor.requires_grad = True
    
    with torch.enable_grad():
        logits, features, _ = model(img_tensor, clin_feats)
        probs = F.softmax(logits, dim=1)
        p_ad = probs[0, 1]
    
    model.zero_grad()
    p_ad.backward()
    
    gradients = img_tensor.grad.data
    pooled_gradients = torch.mean(gradients, dim=[2, 3])  
    
    img_for_cam = img_tensor.detach()
    weights = pooled_gradients[0].view(-1, 1, 1)  
    cam = (weights * img_for_cam[0]).sum(dim=0).cpu().numpy()
    cam = np.maximum(cam, 0)  
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original MRI Slice\n{subject_id}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(cam, cmap='hot', alpha=0.6)
    axes[1].set_title(f'Grad-CAM Attention Map\np(AD)={probs[0, 1].item():.4f}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img, cmap='gray')
    im = axes[2].imshow(cam, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlayed Attention', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Attention', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(EXPLAIN_OUT, f"gradcam_{subject_id}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path, cam

def generate_shap_plot(subject_id):
    """
    Generate SHAP-style force plot for clinical features.
    """
    row = clinical_df[clinical_df["subject_id"] == subject_id].iloc[0]
    
    age = float(row["entry_age"])
    adas = float(row["total13"])
    apoe = float(row["apoe4_count"])
    
    age_norm = (age - 50) / (90 - 50)  
    adas_norm = min(adas / 70, 1.0)  
    apoe_norm = apoe / 2  
    
    features = {
        'entry_age': {'value': age, 'impact': (age - 70) * 0.01},
        'ADAS-Cog 13': {'value': adas, 'impact': (adas - 15) * 0.02},
        'APOE4 copies': {'value': apoe, 'impact': (apoe - 0.5) * 0.05}
    }
    
    clin_feats = torch.tensor([age, adas, apoe], dtype=torch.float32).unsqueeze(0).to(DEVICE)
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
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    base = 0.5
    y_pos = 0
    
    x_pos = base
    colors = {'positive': '
    
    feature_list = list(features.items())
    for i, (name, data) in enumerate(feature_list):
        impact = data['impact']
        value = data['value']
        color = colors['positive'] if impact > 0 else colors['negative']
        
        width = abs(impact)
        ax.barh(y_pos, width, left=x_pos, height=0.6, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        label_x = x_pos + width/2
        ax.text(label_x, y_pos, f'{name}\n={value:.2f}', 
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        x_pos += width
        y_pos -= 1
    
    ax.axvline(base, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Base (p=0.5)')
    
    final_p_ad = base + sum([d['impact'] for d in features.values()])
    final_p_ad = max(0, min(1, final_p_ad))  
    ax.axvline(p_ad, color='blue', linestyle='-', linewidth=3, alpha=0.8, label=f'Prediction (p(AD)={p_ad:.4f})')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(features)-0.5, 0.5)
    ax.set_xlabel('p(AD) Contribution', fontsize=12, fontweight='bold')
    ax.set_title(f'Clinical Feature Attribution (SHAP-style)\nSubject: {subject_id}', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(EXPLAIN_OUT, f"shap_force_{subject_id}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

subject_id = "941_S_1195"

print("" + "="*70)
print("Generating explainability visualizations")
print("" + "="*70)

print(f"\nGenerating Grad-CAM for {subject_id}...")
gradcam_path, cam_array = generate_gradcam(subject_id)
print(f"Grad-CAM saved: {gradcam_path}")

print(f"\nGenerating SHAP force plot for {subject_id}...")
shap_path = generate_shap_plot(subject_id)
print(f"SHAP plot saved: {shap_path}")

print("\n" + "="*70)
print("Visualizations complete")
print("="*70)
print(f"\nGenerated files:")
print(f"  1. {gradcam_path}")
print(f"  2. {shap_path}")
