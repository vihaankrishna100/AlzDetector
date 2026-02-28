import os
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

NIFTI_DIR = "NIFTI_ONE_PER_SUBJECT"

CLINICAL_CSV = "clinical_100_subjects.csv"

class ADNIDataset(Dataset):
    def __init__(self):
        full_df = pd.read_csv(CLINICAL_CSV)

        valid_rows = []
        missing = []

        for _, row in full_df.iterrows():

            subject_id = row["subject_id"]
            pattern = os.path.join(NIFTI_DIR, f"*{subject_id}*.nii.gz")
            
            matches = glob.glob(pattern)
            if len(matches) > 0:
                valid_rows.append(row)
            else:
                missing.append(subject_id)

        self.df = pd.DataFrame(valid_rows)
        self.nifti_dir = NIFTI_DIR

        print(f"Found MRI for {len(self.df)} subjects out of {len(full_df)} in {CLINICAL_CSV}")
        if missing:
            print(f"No MRI found for {len(missing)} subjects (first few: {missing[:5]})")

        if len(self.df) == 0:
            raise RuntimeError("No subjects with matching MRI files. "
                               "Check NIFTI_ONE_PER_SUBJECT and subject_id naming.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["subject_id"]
        label = int(row["label"])

        pattern = os.path.join(self.nifti_dir, f"*{subject_id}*.nii.gz")
        matches = glob.glob(pattern)
        if len(matches) == 0:
            raise FileNotFoundError(f"No MRI for {subject_id} (should not happen if filtered)")

        nii_path = matches[0]
        nii = nib.load(nii_path).get_fdata()

        z = nii.shape[2] // 2
        img = nii[:, :, z]
        
        zoom_factors = (256/img.shape[0], 256/img.shape[1])
        img = ndimage.zoom(img, zoom_factors, order=1)
        if img.shape != (256, 256):
            img = img[:256, :256]
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = torch.tensor(img).unsqueeze(0).float()  

        clin = torch.tensor([
            row["entry_age"],
            row["total13"],
            row["apoe4_count"]
        ]).float()

        return img, clin, label, subject_id
