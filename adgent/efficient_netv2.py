import torch
import torch.nn as nn
import timm


class EffNetV2Clinical(nn.Module):
    def __init__(self, num_classes=2, clin_dim=3):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            in_chans=3,
            num_classes=0,
        )

        mri_dim = self.backbone.num_features

        self.clinical_mlp = nn.Sequential(
            nn.Linear(clin_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(mri_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_mri, x_clin):
        if x_mri.shape[1] == 1:
            x_mri = x_mri.repeat(1, 3, 1, 1)

        feats = self.backbone(x_mri)
        clin_feats = self.clinical_mlp(x_clin)
        fused = torch.cat([feats, clin_feats], dim=1)
        out = self.classifier(fused)

        return out, feats, clin_feats
