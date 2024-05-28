import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerClassifier(nn.Module):
    def __init__(self, input_shape=117, d_model=512, nhead=8, num_layers=3, num_classes=1):
        super(TransformerClassifier, self).__init__()
        self.up_linear = nn.Linear(1,512)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.seq_len = 117
        self.linear1 = nn.Linear(512, 1)
        self.linear2 = nn.Linear(117, 1)
        
        # TransformerEncoder
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Linear layer
        self.fc = nn.Linear(d_model*self.seq_len, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.up_linear(x.permute(0, 2, 1))
        # Reshape input to (seq_len, batch_size, d_model)
        x = x.transpose(0, 1).contiguous()
        x = x.view(self.seq_len, -1, self.d_model)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
  
        x = self.linear1(x)
        x = self.linear2(x.permute(1, 2, 0))
        
        return self.sigmoid(x.squeeze())