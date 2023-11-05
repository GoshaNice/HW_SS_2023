from typing import List

import torch
from torch import Tensor
import numpy as np

from src.base.base_metric import BaseMetric
from src.metric.utils import calc_si_sdr


class SiSDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, batch_prediction: Tensor, batch_target: Tensor, **kwargs
    ):
        si_sdrs = []
        predictions = batch_prediction.cpu().detach().numpy()
        targets = batch_target.cpu().detach().numpy()
        for prediction, target in zip(predictions, targets):
            si_sdr = calc_si_sdr(prediction, target)
            si_sdrs.append(si_sdr)
        return sum(si_sdrs) / len(si_sdrs)