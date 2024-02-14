import torch
from pytorch_lightning import Trainer
from model_lighting import AdditionTransformerEncoderL, AdditionTransformerL
from data.addition import AdditionDataModule


dataloader = AdditionDataModule(batch_size=128, label_type='string')
model = AdditionTransformerL()

# Initialize a trainer
trainer = Trainer(max_epochs=20, accelerator='mps' if torch.backends.mps.is_available() else None)

# Train the model
trainer.fit(model, datamodule=dataloader)

# Save the model to disk (optional)
torch.save(model.state_dict(), 'models/model.pth')
