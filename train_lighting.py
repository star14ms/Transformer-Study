import torch
from torch.nn.functional import one_hot
import math

from pytorch_lightning import Trainer
from model_lighting import TransformerEncoderL
from data.addition import vocab, AdditionDataModule
from device import get_device


dataloader = AdditionDataModule(batch_size=128)


# Initialize your model
model = TransformerEncoderL(
    vocab_size=len(vocab),
    d_model=32,
    nhead=8,
    num_layers=1,
    dim_feedforward=32,
    batch_first=True,
)

# Initialize a trainer
trainer = Trainer(max_epochs=20, accelerator='mps' if torch.backends.mps.is_available() else None)

# Train the model
trainer.fit(model, datamodule=dataloader)

# Save the model to disk (optional)
torch.save(model.state_dict(), 'model0.pth')