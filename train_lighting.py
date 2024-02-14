import torch
from pytorch_lightning import Trainer
from model_lighting import TransformerEncoderL, TransformerL
from data.addition import tokenizer, vocab, AdditionDataModule


dataloader = AdditionDataModule(batch_size=128, label_type='string')

model = TransformerL(
    vocab_size=len(vocab),
    d_model=32,
    nhead=8,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=32,
    batch_first=True,
    PAD_ID=tokenizer.token_to_id('[PAD]'),
)

# Initialize a trainer
trainer = Trainer(max_epochs=20, accelerator='mps' if torch.backends.mps.is_available() else None)

# Train the model
trainer.fit(model, datamodule=dataloader)

# Save the model to disk (optional)
torch.save(model.state_dict(), 'model.pth')
