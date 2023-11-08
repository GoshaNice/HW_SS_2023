import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SpExLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
    
    def crop_or_pad(self, s, target):
        if s.shape == target.shape:
            return s
        if s.shape[-1] > target.shape[-1]:
            return s[:, :target.shape[-1]]
        else:
            return F.pad(
                s,
                (0, int(target.shape[-1] - s.shape[-1])),
                "constant",
                0,
            )

    def forward(
        self, s1, s2, s3, target, probs, target_id=None, **batch
    ) -> Tensor:
        target = target.to(s1.device)
        s1 = s1.squeeze(1)
        s2 = s2.squeeze(1)
        s3 = s3.squeeze(1)
        probs = probs.squeeze(1)
        s1 = self.crop_or_pad(s1, target)
        s2 = self.crop_or_pad(s2, target)
        s3 = self.crop_or_pad(s3, target)

        sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True)
        sisdr = sisdr.to(s1.device)
        loss = torch.zeros((s1.shape[0]), device=s1.device)
        loss -= (1 - self.alpha - self.beta) * sisdr(s1, target)
        loss -= self.alpha * sisdr(s2, target)
        loss -= self.beta * sisdr(s3, target)
        if target_id is not None:
            probs = torch.softmax(probs, dim=1)
            self.ce = self.ce.to(probs.device)
            loss += self.gamma * self.ce(probs, target_id)
        return loss.mean()
