import torch
from torch import nn
import math
from transformers import AutoTokenizer


class PositionalEncoding(torch.nn.Module):
  def __init__(self, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(5000, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / torch.pow(10000, (2 * (i // 2)) / torch.FloatTensor([d_model]))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=torch.arange(position).unsqueeze(1),
        i=torch.arange(d_model).unsqueeze(0),
        d_model=d_model)

    # apply the sine function to the even indices of the array (2i)
    sines = torch.sin(angle_rads[:, 0::2])

    # apply the cosine function to the odd indices of the array (2i+1)
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    pos_encoding = pos_encoding.unsqueeze(0)

    print(pos_encoding.shape)
    return pos_encoding.float()

  def forward(self, inputs):
    return inputs + self.pos_encoding[:, :inputs.shape[1], :]


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab = tokenizer.get_vocab()
 
    ntokens = len(vocab) # size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
    src = torch.randint(0, ntokens, (10, 32)) # 10 tokens in a batch of 32
    out = model(src)

    print(out.shape)
