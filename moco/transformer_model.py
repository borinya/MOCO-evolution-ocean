import torch
import torch.nn as nn
#adwda
class OceanTransformer(nn.Module):
    def __init__(self, input_dim=256, num_layers=4, nhead=8, 
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        x = self.transformer(x)
        return self.output_proj(x[:, -1, :])