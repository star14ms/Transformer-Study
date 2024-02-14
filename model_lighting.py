import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import one_hot
from model import TransformerEncoder, Transformer


class TransformerEncoderL(pl.LightningModule):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, batch_first, *args, **kwargs):
        super().__init__()
        self.model = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, inputs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs)

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.abs(outputs - labels).mean() ** 2
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class TransformerL(pl.LightningModule):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first, PAD_ID, *args, **kwargs):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.PAD_ID = PAD_ID

        # self.src_mask = nn.Transformer.generate_square_subsequent_mask(7)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(5)
        self.memory_mask = nn.Transformer.generate_square_subsequent_mask(7)

    def forward(self, inputs, labels_shift):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, labels_shift)

    def training_step(self, batch):
        inputs, labels_shift, labels = batch

        kwargs = {
            # 'src_mask': self.src_mask,
            'tgt_mask': self.tgt_mask.to(self.device),
            # 'memory_mask': self.memory_mask,
            'src_key_padding_mask': (inputs == self.PAD_ID).to(torch.float32).to(self.device),
            'tgt_key_padding_mask': (labels == self.PAD_ID).to(torch.float32).to(self.device),
            # 'memory_key_padding_mask': (inputs == self.PAD_ID).to(torch.float32).to(self.device),
            # 'tgt_is_causal': True,
            # 'src_is_causal': True,
            # 'memory_is_causal': True,
        }

        # forward + backward + optimize
        outputs = self.model(inputs, labels_shift, **kwargs)
        outputs_prob = torch.softmax(outputs, dim=2)
        labels = one_hot(labels, num_classes=self.vocab_size).to(torch.float32)
        loss = self.criterion(outputs_prob, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer