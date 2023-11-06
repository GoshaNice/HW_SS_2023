import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def calc_si_sdr(est: torch.Tensor, target: torch.Tensor):
    """Calculate SI-SDR metric for two given tensors"""
    assert est.shape == target.shape, "Input and Target should have the same shape"
    alpha = (target * est).sum(dim=-1) / torch.norm(target, dim=-1)**2
    return 20 * torch.log10(torch.norm(alpha.unsqueeze(1) * target, dim=-1) / (torch.norm(alpha.unsqueeze(1) * target - est, dim=-1) + 1e-6) + 1e-6)

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
        loss = torch.zeros((s1.shape[0]), device=s1.device)
        loss -= (1 - self.alpha - self.beta) * calc_si_sdr(s1, target)
        loss -= self.alpha * calc_si_sdr(s2, target)
        loss -= self.beta * calc_si_sdr(s3, target)
        if target_id is not None:
            probs = torch.softmax(probs, dim=1)
            self.ce = self.ce.to(probs.device)
            loss += self.gamma * self.ce(probs, target_id)
        return loss.mean()
