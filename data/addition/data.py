import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from pytorch_lightning import LightningDataModule


class AdditionDataset(Dataset):
    def __init__(self, txt_file, label_type='string', transform=None, max_length=7):
        with open(txt_file, 'r') as f:
          data = f.readlines()

        self.data = data
        self.transform = transform
        self.max_length = max_length
        self.label_type = label_type
        self.SOS_ID = tokenizer.token_to_id('[SOS]')
        self.EOS_ID = tokenizer.token_to_id('[EOS]')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs, labels = self.data[idx].split('_')
        inputs = tokenizer.encode(inputs).ids
        if self.label_type == 'string':
            labels = tokenizer.encode(labels.replace('\n', '')).ids
            labels_shift = [self.SOS_ID] + labels
            labels = labels + [self.EOS_ID]
        elif self.label_type == 'int':
            labels = [int(labels.replace('\n', ''))]

        if self.transform:
            inputs = self.transform(inputs)
            labels = self.transform(labels)
            if self.label_type == 'string':
                labels_shift = self.transform(labels_shift)

        if self.label_type == 'string':
            return inputs, labels_shift, labels
        return inputs, labels


class AdditionDataModule(LightningDataModule):
    def __init__(self, batch_size=128, label_type='string'):
        super().__init__()
        self.batch_size = batch_size
        self.label_type = label_type

    def train_dataloader(self, num_workers=0):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor([int(i) for i in x], dtype=torch.long)),
        ])

        dataset = AdditionDataset(txt_file='data/addition/addition.txt', label_type=self.label_type, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        return dataloader
    

tokenizer = Tokenizer.from_file("data/addition/tokenizer.json")
vocab = tokenizer.get_vocab()


if __name__ == '__main__':
    label_type = 'string'
    datamodule = AdditionDataModule(batch_size=1, label_type=label_type)
    dataloader = datamodule.train_dataloader()

    for i, data in enumerate(dataloader):
        if label_type == 'string':
            inputs, _, labels = data
            print(inputs.shape, labels.shape)
            print(tokenizer.decode(inputs[0].tolist()).replace(' ', ''))
            print(tokenizer.decode(labels[0].tolist()).replace(' ', ''))
        elif label_type == 'int':
            inputs, labels = data
            print(inputs.shape, labels.shape)
            print(tokenizer.decode(inputs[0].tolist()))
            print(labels[0].tolist()[0])
        break
