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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs, labels = self.data[idx].split('_')
        inputs = tokenizer.encode(inputs).ids
        if self.label_type == 'string':
            labels = tokenizer.encode(labels.replace('\n', '')).ids
        elif self.label_type == 'int':
            labels = [int(labels.replace('\n', ''))]

        if self.transform:
            inputs = self.transform(inputs)
            labels = self.transform(labels)

        return inputs, labels


class AdditionDataModule(LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, label_type='int', num_workers=0):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor([int(i) for i in x], dtype=torch.long)),
        ])

        dataset = AdditionDataset(txt_file='data/addition/addition.txt', label_type=label_type, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        return dataloader
    

tokenizer = Tokenizer.from_file("data/addition/tokenizer.json")
vocab = tokenizer.get_vocab()


if __name__ == '__main__':
    label_type = 'int'
    dataloader = AdditionDataModule.train_dataloader(label_type=label_type)

    for i, data in enumerate(dataloader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        print(tokenizer.decode(inputs[0].tolist()))

        if label_type == 'string':
            print(tokenizer.decode(labels[0].tolist()))
        elif label_type == 'int':
            print(labels[0].tolist()[0])
        break
