from torch import Tensor

from src.base.base_metric import BaseMetric
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


class PESQMetric(BaseMetric):
    def __init__(self, fs=8000, mode="nb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(
        self, prediction: Tensor, target: Tensor, **kwargs
    ):
        prediction = prediction.squeeze(1)
        pesq = self.pesq(prediction, target)
        return pesq.mean()