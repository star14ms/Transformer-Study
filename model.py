from torch import nn
import torch


class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()

        self.decoderlayer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers=2, norm=nn.LayerNorm(512))
        self.ffn = nn.Linear(512, 10)

    def forward(self, x):
        x = self.decoder(x)
        x = self.ffn(x)

        return x
    

class TransformerEncoder_forMNIST(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, num_classes=10):
        super(TransformerEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.ffn = nn.Linear(18432, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x.view(x.shape[0], x.shape[1], -1)
        # x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.ffn(x)

        return x.view(x.shape[0], -1)
