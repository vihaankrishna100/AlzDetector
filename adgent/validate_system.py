"""
Validation script to diagnose blockers and check system readiness.
Run this to identify what needs to be fixed before running training/inference.
"""
#AI CREATED TEST CASE AND VALIDATION
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

def check(condition, message_pass, message_fail):
    """Print pass/fail message."""
    if condition:
        print(f"{message_pass}")
        return True
    else:
        print(f"{message_fail}")
        return False

def main():
    print("\n" + "="*70)
    print("ADgent Framework - System Readiness Check")
    print("="*70 + "\n")
    
    all_pass = True
    
    print("DATA FILES CHECK")
    all_pass &= check(
        os.path.exists("clinical_100_subjects.csv"),
        "clinical_100_subjects.csv found",
        "clinical_100_subjects.csv MISSING"
    )
    
    if os.path.exists("clinical_100_subjects.csv"):
        
        df = pd.read_csv("clinical_100_subjects.csv")
        required_cols = {"subject_id", "label", "entry_age", "total13", "apoe4_count"}
        actual_cols = set(df.columns)
        has_cols = required_cols.issubset(actual_cols)
        all_pass &= check(
            has_cols,
            f"CSV has correct columns: {sorted(required_cols)}",
            f"CSV missing columns! Has: {sorted(actual_cols)}, Needs: {sorted(required_cols)}"
        )
        print(f"   → Found {len(df)} subjects in CSV")
    
    print("\nMRI DATA CHECK")
    nifti_dir = "NIFTI_ONE_PER_SUBJECT"
    nifti_exists = os.path.isdir(nifti_dir)
    all_pass &= check(
        nifti_exists,
        f"Directory {nifti_dir} exists",
        f"Directory {nifti_dir} MISSING"
    )
    
    if nifti_exists:
        nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
        all_pass &= check(
            len(nifti_files) > 0,
            f"Found {len(nifti_files)} .nii.gz files in {nifti_dir}",
            f"No .nii.gz files in {nifti_dir} - THIS BLOCKS TRAINING"
        )
    
    print("\nMODEL CHECK")
    model_exists = os.path.exists("model.pth")
    all_pass &= check(
        model_exists,
        "model.pth exists (already trained)",
        "model.pth MISSING - Must run: python train.py"
    )
    
    print("\nPACKAGE CHECK")
    packages_to_check = [
        ("torch", "PyTorch"),
        ("nibabel", "NiBabel"),
        ("pandas", "Pandas"),
        ("crewai", "CrewAI"),
        ("timm", "TIMM (EfficientNetV2)"),
        ("shap", "SHAP"),
    ]
    
    for pkg, name in packages_to_check:
        try:
            __import__(pkg)
            check(True, f"{name} installed", "")
        except ImportError:
            all_pass &= check(False, "", f"{name} NOT installed - Run: pip install {pkg}")
    
    print("\nLLM CONFIGURATION CHECK")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    has_llm = bool(openai_key or anthropic_key)
    
    if openai_key:
        print(f"OPENAI_API_KEY configured (found)")
    elif anthropic_key:
        print(f"ANTHROPIC_API_KEY configured (found)")
    else:
        print(f"No LLM API key found - CrewAI agents need OpenAI or Anthropic key")
        print(f"   Set: export OPENAI_API_KEY='...' or ANTHROPIC_API_KEY='...'")
        all_pass = False
    
    print("\nAGENT FILES CHECK")
    agent_files = [
        "agents/__init__.py",
        "agents/agents.py",
        "agents/tools.py",
        "agents/crew_runner.py",
        "agents/main.py",
    ]
    for f in agent_files:
        all_pass &= check(
            os.path.exists(f),
            f"{f} exists",
            f"{f} MISSING"
        )
    
    print("\n" + "="*70)
    if all_pass:
        print("ALL CHECKS PASSED - Ready to proceed!")
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Run: python -m agents.main")
    else:
        print("SOME CHECKS FAILED - See above for details")
        print("\nCritical blockers to fix first:")
        print("  • MRI files in NIFTI_ONE_PER_SUBJECT")
        print("  • Model training (python train.py)")
        print("  • LLM API key for CrewAI agents")
    print("="*70 + "\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
