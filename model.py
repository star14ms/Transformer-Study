from torch import nn
import math
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]

        return self.dropout(x)


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
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, batch_first=False):
        super(Transformer, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=7)
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt):
        x = self.embbeding(x)
        x = self.pos_encoding(x)
        tgt = self.embbeding(tgt)
        tgt = self.pos_encoding(tgt)
        x = self.transformer(x, tgt)
        x = self.linear(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=7)
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model*7, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.embbeding(x)
        h = self.pos_encoding(x)
        h = self.transformer(h)
        h = x + h
        h = h.view(h.shape[0], -1)
        h = self.ffn(h)

        return h
    

class LinearEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model*7, 64)
        self.ffn = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.embbeding(x)
        h = x.view(x.shape[0], -1)
        h = self.linear(h)
        h = self.ffn(h)

        return h