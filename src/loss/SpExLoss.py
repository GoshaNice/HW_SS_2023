import torch
from torch import Tensor
from torch import nn
#from src.metric.utils import calc_si_sdr


def calc_si_sdr(est: torch.Tensor, target: torch.Tensor):
    """Calculate SI-SDR metric for two given tensors"""
    assert est.shape == target.shape, "Input and Target should have the same shape"
    degrade = est - target
    si_sdr = (target * degrade).sum(axis=-1) / ((target ** 2 + degrade ** 2) // 2)
    return si_sdr


class SpExLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
    def forward(
        self, s1, s2, s3, target, probs, speaker_id, **batch
    ) -> Tensor:
        loss = torch.zeros((s1.shape[0], 1))
        loss += self.alpha * calc_si_sdr(s1, target)
        loss += self.beta * calc_si_sdr(s2, target)
        loss += self.beta * calc_si_sdr(s3, target)
        loss += self.gamma * self.ce(probs, speaker_id)
        return loss
