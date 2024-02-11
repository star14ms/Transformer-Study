import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import one_hot
from model import TransformerEncoder


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
