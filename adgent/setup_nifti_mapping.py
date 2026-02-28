# maps nifti file to the subject id
#Mainly AI created test(slight errors)
import os
import glob
import pandas as pd

NIFTI_DIR = "NIFTI_ONE_PER_SUBJECT"

#this is the clinical data for the 100 subjects
CLINICAL_CSV = "clinical_100_subjects.csv"

df = pd.read_csv(CLINICAL_CSV)
nifti_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*.nii.gz")))


print(f"Found {len(nifti_files)} NIfTI files")
print(f"Found {len(df)} subjects in clinical data")

for i, (subject_id, nifti_file) in enumerate(zip(df["subject_id"], nifti_files)):
    old_name = nifti_file
    basename = os.path.basename(nifti_file)
    new_basename = f"{subject_id}_{basename}"
    new_path = os.path.join(NIFTI_DIR, new_basename)
    
    if not os.path.exists(new_path):
        try:
            os.symlink(basename, new_path)
            print(f" {i+1}/{len(df)} Linked: {subject_id}")
        except Exception as e:
            print(f"  {i+1}/{len(df)} Error: {e}")
    else:
        print(f"  {i+1}/{len(df)} Already exists: {new_basename}")

print("\n Setup complete! You can now run train.py")
