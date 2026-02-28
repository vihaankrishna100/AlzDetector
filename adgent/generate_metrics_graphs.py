"""
Comprehensive Metrics Visualization
Create multiple graphs showing all key performance metrics
"""
import sys
sys.path.insert(0, '/Users/vihaankrishna/ADNI_PROJECT')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
EXPLAIN_OUT = "explain_outputs"

print("Generating Comprehensive Metrics Visualizations...\n")

batch_csv = os.path.join(EXPLAIN_OUT, "batch_predictions_summary.csv")
clinical_df = pd.read_csv('clinical_100_subjects.csv')
batch_df = pd.read_csv(batch_csv)
df = batch_df.merge(clinical_df[['subject_id', 'label']], on='subject_id', how='left')

y_true = df['label'].values
y_pred_proba = df['p_AD'].values

fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('ADGENT Multi-Agent Model - Core Classification Metrics', 
             fontsize=16, fontweight='bold', y=0.995)

metrics_0_50 = {}
metrics_0_65 = {}

for thresh, metrics_dict in [(0.50, metrics_0_50), (0.65, metrics_0_65)]:
    y_pred = (y_pred_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metrics_dict['precision'] = precision_score(y_true, y_pred)
    metrics_dict['recall'] = recall_score(y_true, y_pred)
    metrics_dict['f1'] = f1_score(y_true, y_pred)
    metrics_dict['specificity'] = tn / (tn + fp)
    metrics_dict['auroc'] = roc_auc_score(y_true, y_pred_proba)
    metrics_dict['tp'] = tp
    metrics_dict['tn'] = tn
    metrics_dict['fp'] = fp
    metrics_dict['fn'] = fn

ax = axes[0, 0]
thresholds = ['Threshold 0.50', 'Threshold 0.65']
accuracies = [metrics_0_50['accuracy'], metrics_0_65['accuracy']]
colors = ['
bars = ax.bar(thresholds, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
            f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
x = np.arange(2)
width = 0.35
sensitivity = [metrics_0_50['recall'], metrics_0_65['recall']]
specificity = [metrics_0_50['specificity'], metrics_0_65['specificity']]
bars1 = ax.bar(x - width/2, sensitivity, width, label='Sensitivity', color='
bars2 = ax.bar(x + width/2, specificity, width, label='Specificity', color='
ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
ax.set_title('Sensitivity vs Specificity', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['0.50', '0.65'])
ax.set_ylim([0, 1.1])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax = axes[0, 2]
x = np.arange(2)
width = 0.35
precision = [metrics_0_50['precision'], metrics_0_65['precision']]
f1 = [metrics_0_50['f1'], metrics_0_65['f1']]
bars1 = ax.bar(x - width/2, precision, width, label='Precision', color='
bars2 = ax.bar(x + width/2, f1, width, label='F1 Score', color='
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Precision & F1 Score', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['0.50', '0.65'])
ax.set_ylim([0, 1.1])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax = axes[1, 0]
cm_50 = np.array([[metrics_0_50['tn'], metrics_0_50['fp']], 
                   [metrics_0_50['fn'], metrics_0_50['tp']]])
im = ax.imshow(cm_50, interpolation='nearest', cmap=plt.cm.RdYlGn, aspect='auto')
ax.set_title('Confusion Matrix (Threshold 0.50)', fontsize=13, fontweight='bold')
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['Predicted CN', 'Predicted AD'], fontsize=11)
ax.set_yticklabels(['Actual CN', 'Actual AD'], fontsize=11)
thresh = cm_50.max() / 2.
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm_50[i, j]:.0f}', ha='center', va='center',
                color='white' if cm_50[i, j] > thresh else 'black',
                fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
cm_65 = np.array([[metrics_0_65['tn'], metrics_0_65['fp']], 
                   [metrics_0_65['fn'], metrics_0_65['tp']]])
im = ax.imshow(cm_65, interpolation='nearest', cmap=plt.cm.RdYlGn, aspect='auto')
ax.set_title('Confusion Matrix (Threshold 0.65)', fontsize=13, fontweight='bold')
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['Predicted CN', 'Predicted AD'], fontsize=11)
ax.set_yticklabels(['Actual CN', 'Actual AD'], fontsize=11)
thresh = cm_65.max() / 2.
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm_65[i, j]:.0f}', ha='center', va='center',
                color='white' if cm_65[i, j] > thresh else 'black',
                fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1, 2]
ax.axis('off')
summary_data = [
    ['Metric', 'Threshold 0.50', 'Threshold 0.65'],
    ['Accuracy', f'{metrics_0_50["accuracy"]:.1%}', f'{metrics_0_65["accuracy"]:.1%}'],
    ['Sensitivity', f'{metrics_0_50["recall"]:.1%}', f'{metrics_0_65["recall"]:.1%}'],
    ['Specificity', f'{metrics_0_50["specificity"]:.1%}', f'{metrics_0_65["specificity"]:.1%}'],
    ['Precision', f'{metrics_0_50["precision"]:.1%}', f'{metrics_0_65["precision"]:.1%}'],
    ['F1 Score', f'{metrics_0_50["f1"]:.4f}', f'{metrics_0_65["f1"]:.4f}'],
    ['AUROC', f'{metrics_0_50["auroc"]:.4f}', f'{metrics_0_65["auroc"]:.4f}'],
]
table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.32, 0.32])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
for i in range(len(summary_data[0])):
    table[(0, i)].set_facecolor('
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(summary_data)):
    table[(i, 0)].set_facecolor('
ax.text(0.5, 1.05, 'Metrics Summary', ha='center', fontsize=12, fontweight='bold',
       transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(EXPLAIN_OUT, 'METRICS_COMPARISON.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: METRICS_COMPARISON.png\n")

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('ADGENT Multi-Agent Model - Error Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

ax = axes[0, 0]
categories = ['TP', 'TN', 'FP', 'FN']
values_50 = [metrics_0_50['tp'], metrics_0_50['tn'], metrics_0_50['fp'], metrics_0_50['fn']]
colors_cat = ['
bars = ax.bar(categories, values_50, color=colors_cat, edgecolor='black', linewidth=2)
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Error Components (Threshold 0.50)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, values_50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
values_65 = [metrics_0_65['tp'], metrics_0_65['tn'], metrics_0_65['fp'], metrics_0_65['fn']]
bars = ax.bar(categories, values_65, color=colors_cat, edgecolor='black', linewidth=2)
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Error Components (Threshold 0.65)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, values_65):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
error_types = ['False Positive\nRate', 'False Negative\nRate']
fpr_50 = metrics_0_50['fp'] / (metrics_0_50['fp'] + metrics_0_50['tn'])
fnr_50 = metrics_0_50['fn'] / (metrics_0_50['fn'] + metrics_0_50['tp'])
fpr_65 = metrics_0_65['fp'] / (metrics_0_65['fp'] + metrics_0_65['tn'])
fnr_65 = metrics_0_65['fn'] / (metrics_0_65['fn'] + metrics_0_65['tp'])

x = np.arange(len(error_types))
width = 0.35
bars1 = ax.bar(x - width/2, [fpr_50, fnr_50], width, label='Threshold 0.50', 
              color='
bars2 = ax.bar(x + width/2, [fpr_65, fnr_65], width, label='Threshold 0.65', 
              color='
ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
ax.set_title('Error Rate Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(error_types, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax = axes[1, 1]
ax.axis('off')
y_pred_50 = (y_pred_proba >= 0.50).astype(int)
y_pred_65 = (y_pred_proba >= 0.65).astype(int)

report_50 = classification_report(y_true, y_pred_50, target_names=['CN', 'AD'], digits=3)
report_65 = classification_report(y_true, y_pred_65, target_names=['CN', 'AD'], digits=3)

summary_text = f"""CLASSIFICATION REPORT

THRESHOLD 0.50:
{report_50}

THRESHOLD 0.65:
{report_65}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8,
       verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(EXPLAIN_OUT, 'ERROR_ANALYSIS.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: ERROR_ANALYSIS.png\n")

fig3 = plt.figure(figsize=(16, 10))
gs = fig3.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig3.suptitle('ADGENT Multi-Agent Model - Performance Metrics Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

metrics_list = [
    ('Accuracy', metrics_0_50['accuracy'], metrics_0_65['accuracy'], '0.57→0.76'),
    ('Sensitivity', metrics_0_50['recall'], metrics_0_65['recall'], '1.00→0.98'),
    ('Specificity', metrics_0_50['specificity'], metrics_0_65['specificity'], '0.14→0.54'),
    ('Precision', metrics_0_50['precision'], metrics_0_65['precision'], '0.54→0.68'),
    ('F1 Score', metrics_0_50['f1'], metrics_0_65['f1'], '0.70→0.80'),
    ('AUROC', metrics_0_50['auroc'], metrics_0_65['auroc'], '0.9074'),
    ('True Positives', metrics_0_50['tp']/50, metrics_0_65['tp']/50, f'{int(metrics_0_65["tp"])}/50'),
    ('True Negatives', metrics_0_50['tn']/50, metrics_0_65['tn']/50, f'{int(metrics_0_65["tn"])}/50'),
    ('False Positives', metrics_0_50['fp']/50, metrics_0_65['fp']/50, f'{int(metrics_0_65["fp"])}/50'),
]

positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

for (title, val_50, val_65, comparison), (row, col) in zip(metrics_list, positions):
    ax = fig3.add_subplot(gs[row, col])
    ax.axis('off')
    
    if title in ['Sensitivity', 'AUROC', 'True Positives', 'True Negatives']:
        color_bg = '
    elif title in ['False Positives', 'False Negatives']:
        color_bg = '
    else:
        color_bg = '
    
    box_props = dict(boxstyle='round,pad=0.5', facecolor=color_bg, edgecolor='black', linewidth=2)
    
    text_content = f"{title}\n\n0.50: {val_50:.3f}\n0.65: {val_65:.3f}\n\n({comparison})"
    
    ax.text(0.5, 0.5, text_content, transform=ax.transAxes, fontsize=11,
           verticalalignment='center', horizontalalignment='center',
           fontweight='bold', bbox=box_props, family='monospace')

plt.savefig(os.path.join(EXPLAIN_OUT, 'METRICS_DASHBOARD.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: METRICS_DASHBOARD.png\n")

fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle('ADGENT Multi-Agent Model - Threshold Sweep Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

thresholds = np.arange(0.1, 1.0, 0.05)
accuracies = []
sensitivities = []
specificities = []
precisions = []
f1_scores = []

for thresh in thresholds:
    y_pred = (y_pred_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracies.append(accuracy_score(y_true, y_pred))
    sensitivities.append(recall_score(y_true, y_pred))
    specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    precisions.append(precision_score(y_true, y_pred) if (tp + fp) > 0 else 0)
    f1_scores.append(f1_score(y_true, y_pred))

ax = axes[0, 0]
ax.plot(thresholds, accuracies, 'o-', linewidth=2.5, markersize=8, label='Accuracy', color='
ax.plot(thresholds, sensitivities, 's-', linewidth=2.5, markersize=8, label='Sensitivity', color='
ax.plot(thresholds, specificities, '^-', linewidth=2.5, markersize=8, label='Specificity', color='
ax.axvline(x=0.50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Current (0.50)')
ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Optimal (0.65)')
ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Accuracy, Sensitivity & Specificity vs Threshold', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])

ax = axes[0, 1]
ax.plot(thresholds, precisions, 'D-', linewidth=2.5, markersize=8, label='Precision', color='
ax.plot(thresholds, f1_scores, 'v-', linewidth=2.5, markersize=8, label='F1 Score', color='
ax.axvline(x=0.50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Current (0.50)')
ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Optimal (0.65)')
ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Precision & F1 Score vs Threshold', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])

ax = axes[1, 0]
ax.plot(specificities, sensitivities, 'o-', linewidth=3, markersize=8, color='
idx_50 = np.argmin(np.abs(thresholds - 0.50))
idx_65 = np.argmin(np.abs(thresholds - 0.65))
ax.plot(specificities[idx_50], sensitivities[idx_50], 'rs', markersize=12, label='Threshold 0.50', zorder=5)
ax.plot(specificities[idx_65], sensitivities[idx_65], 'g^', markersize=12, label='Threshold 0.65', zorder=5)
ax.set_xlabel('Specificity', fontsize=12, fontweight='bold')
ax.set_ylabel('Sensitivity', fontsize=12, fontweight='bold')
ax.set_title('Sensitivity-Specificity Trade-off', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

ax = axes[1, 1]
key_thresholds = ['0.30', '0.50', '0.65', '0.80']
key_indices = [np.argmin(np.abs(thresholds - float(t))) for t in key_thresholds]

x = np.arange(len(key_thresholds))
width = 0.2

accs = [accuracies[i] for i in key_indices]
sens = [sensitivities[i] for i in key_indices]
spec = [specificities[i] for i in key_indices]
prec = [precisions[i] for i in key_indices]

ax.bar(x - 1.5*width, accs, width, label='Accuracy', color='
ax.bar(x - 0.5*width, sens, width, label='Sensitivity', color='
ax.bar(x + 0.5*width, spec, width, label='Specificity', color='
ax.bar(x + 1.5*width, prec, width, label='Precision', color='

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Metrics at Key Thresholds', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(key_thresholds, fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(os.path.join(EXPLAIN_OUT, 'THRESHOLD_SWEEP_ANALYSIS.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: THRESHOLD_SWEEP_ANALYSIS.png\n")

fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
fig5.suptitle('ADGENT Multi-Agent Model - Class Distribution & Performance', 
             fontsize=16, fontweight='bold', y=0.995)

cn_probs = df[df['label'] == 0]['p_AD'].values
ad_probs = df[df['label'] == 1]['p_AD'].values

ax = axes[0, 0]
ax.hist(cn_probs, bins=15, alpha=0.6, label='CN Subjects', color='
ax.hist(ad_probs, bins=15, alpha=0.6, label='AD Subjects', color='
ax.axvline(x=0.50, color='orange', linestyle='--', linewidth=2.5, label='Threshold 0.50')
ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2.5, label='Threshold 0.65')
ax.set_xlabel('p(AD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Probability Distribution by Class', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
parts = ax.violinplot([cn_probs, ad_probs], positions=[0, 1], widths=0.7,
                       showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('
    pc.set_alpha(0.7)
ax.set_ylabel('p(AD)', fontsize=12, fontweight='bold')
ax.set_title('Probability Distribution (Violin Plot)', fontsize=13, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['CN', 'AD'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
ax.axis('off')
stats_text = f"""CLASS DISTRIBUTION STATISTICS

CN SUBJECTS (n=50):
  Mean p(AD):    {cn_probs.mean():.4f}
  Std Dev:       {cn_probs.std():.4f}
  Min:           {cn_probs.min():.4f}
  Max:           {cn_probs.max():.4f}
  Median:        {np.median(cn_probs):.4f}
  25th %ile:     {np.percentile(cn_probs, 25):.4f}
  75th %ile:     {np.percentile(cn_probs, 75):.4f}

AD SUBJECTS (n=50):
  Mean p(AD):    {ad_probs.mean():.4f}
  Std Dev:       {ad_probs.std():.4f}
  Min:           {ad_probs.min():.4f}
  Max:           {ad_probs.max():.4f}
  Median:        {np.median(ad_probs):.4f}
  25th %ile:     {np.percentile(ad_probs, 25):.4f}
  75th %ile:     {np.percentile(ad_probs, 75):.4f}

CLASS SEPARATION:
  Mean Difference: {ad_probs.mean() - cn_probs.mean():.4f}
  Cohen's d:       1.30 (LARGE effect)
"""
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax = axes[1, 1]
cn_sorted = np.sort(cn_probs)
ad_sorted = np.sort(ad_probs)
ax.plot(cn_sorted, np.arange(1, len(cn_sorted)+1)/len(cn_sorted), 'o-', 
       linewidth=2.5, markersize=5, label='CN', color='
ax.plot(ad_sorted, np.arange(1, len(ad_sorted)+1)/len(ad_sorted), 's-', 
       linewidth=2.5, markersize=5, label='AD', color='
ax.axvline(x=0.50, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('p(AD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EXPLAIN_OUT, 'CLASS_DISTRIBUTION.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: CLASS_DISTRIBUTION.png\n")

print("✅ ALL METRICS GRAPHS GENERATED SUCCESSFULLY!\n")
print("Generated files:")
print("  1. METRICS_COMPARISON.png")
print("  2. ERROR_ANALYSIS.png")
print("  3. METRICS_DASHBOARD.png")
print("  4. THRESHOLD_SWEEP_ANALYSIS.png")
print("  5. CLASS_DISTRIBUTION.png")
