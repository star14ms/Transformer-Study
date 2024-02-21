from torch import nn
import math
import torch
from torch_custom.model import TransformerEncoderLayer, TransformerDecoderLayer


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


class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=7)
        self.pos_encoding_tgt = PositionalEncoding(d_model, max_len=5)
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x, tgt, **kwargs):
        x = self.embbeding(x)
        x = self.pos_encoding(x)
        tgt = self.embbeding(tgt)
        tgt = self.pos_encoding_tgt(tgt)
        x = self.transformer(x, tgt, **kwargs)
        x = self.linear(x)

        return x


class AdditionTransformerV2(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, batch_first=False, residual=True):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=7)
        self.pos_encoding_tgt = PositionalEncoding(d_model, max_len=5)
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                batch_first=batch_first,
                residual=residual
            )
        , num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                batch_first=batch_first,
                residual=residual
            )
        , num_layers=num_decoder_layers)
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x, tgt, **kwargs):
        def _get_encoder_kwargs(kwargs: dict):
            return {
                'mask': kwargs.get('src_mask'),
                'src_key_padding_mask': kwargs.get('src_key_padding_mask'),
                'is_causal': kwargs.get('src_is_causal', False),
            }

        def _get_decoder_kwargs(kwargs: dict):
            return {
                'tgt_mask': kwargs.get('tgt_mask'),
                'memory_mask': kwargs.get('memory_mask'),
                'tgt_key_padding_mask': kwargs.get('tgt_key_padding_mask'),
                'memory_key_padding_mask': kwargs.get('memory_key_padding_mask'),
                'tgt_is_causal': kwargs.get('tgt_is_causal', False),
                'memory_is_causal': kwargs.get('memory_is_causal', False),
            }
        
        x = self.embbeding(x)
        x = self.pos_encoding(x)
        tgt = self.embbeding(tgt)
        tgt = self.pos_encoding_tgt(tgt)
        memory = self.encoder(x, **_get_encoder_kwargs(kwargs))
        x = self.decoder(tgt, memory, **_get_decoder_kwargs(kwargs))
        x = self.linear(x)

        return x


class AdditionTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=7)
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model*7, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, **kwargs):
        def _get_encoder_kwargs(kwargs: dict):
            return {
                'mask': kwargs.get('src_mask'),
                'src_key_padding_mask': kwargs.get('src_key_padding_mask'),
                'is_causal': kwargs.get('src_is_causal', False),
            }

        x = self.embbeding(x)
        h = self.pos_encoding(x)
        h = self.encoder(h, **_get_encoder_kwargs(kwargs))
        h = h.view(h.shape[0], -1)
        h = self.ffn(h)

        return h
    

class AdditionLinearEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512):
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
    

class DateTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=29)
        self.pos_encoding_tgt = PositionalEncoding(d_model, max_len=11)
        self.embbeding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x, tgt, **kwargs):
        x = self.embbeding(x)
        x = self.pos_encoding(x)
        tgt = self.embbeding(tgt)
        tgt = self.pos_encoding_tgt(tgt)
        x = self.transformer(x, tgt, **kwargs)
        x = self.linear(x)

        return x
