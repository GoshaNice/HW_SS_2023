from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

class DeepSpeech2(BaseModel):
    def __init__(self, 
                 input_dim: int, 
                 n_class: int,
                 gru_hidden: int,
                 **batch):
        super().__init__(input_dim, n_class, **batch)
        self.conv = Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, padding = 4, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=input_dim),
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, padding = 4, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=input_dim),
        )
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=3, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=gru_hidden * 2)
        self.fc = nn.Linear(in_features=gru_hidden * 2, out_features=n_class)


    def forward(self, spectrogram, **batch):
        x = self.conv(spectrogram)
        x = x.transpose(1, 2)
        x = self.rnn(x)
        x = x[0].transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here