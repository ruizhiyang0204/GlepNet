import torch
import torch.nn as nn
from components.conformer_encoder import ConformerEncoder

class EEG_Conformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_up = nn.Linear(1, 512)
        self.encoder = ConformerEncoder(num_layers=3, d_model=512, d_ffn=512, nhead=8)
        self.linear1 = nn.Linear(512, 1)
        self.linear2 = nn.Linear(124, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_up(x.permute(0, 2, 1))
        x, _ = self.encoder(x)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        x = self.sigmoid(x)
        return x.squeeze()
