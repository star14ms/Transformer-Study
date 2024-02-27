import torch
from pytorch_lightning import Trainer
from model_lighting import AdditionTransformerEncoderL, AdditionTransformerL, DateTransformerL # choose the model you want to train
from data.addition import AdditionDataModule
from data.date import DateDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


model = DateTransformerL() # change the model
label_type = 'int' if model.__class__.__name__ == 'AdditionTransformerEncoderL' else 'string'

if 'addition' in model.__class__.__name__.lower(): 
    dataloader = AdditionDataModule(batch_size=128, label_type=label_type)
elif 'date' in model.__class__.__name__.lower():
    dataloader = DateDataModule(batch_size=128)

# Initialize a trainer
logger = TensorBoardLogger("./lightning_logs/", name=model.__class__.__name__)
trainer = Trainer(max_epochs=10, logger=logger, accelerator='mps' if torch.backends.mps.is_available() else None)

# Train the model
trainer.fit(model, datamodule=dataloader)

# Save the model to disk (optional)
torch.save(model.state_dict(), 'models/model_date.pth')
