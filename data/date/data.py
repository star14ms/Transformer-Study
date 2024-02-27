import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from pytorch_lightning import LightningDataModule


class DateDataset(Dataset):
    def __init__(self, txt_file, transform=None, max_length=29):
        with open(txt_file, 'r') as f:
          data = f.readlines()

        self.data = data
        self.transform = transform
        self.max_length = max_length
        self.SOS_ID = tokenizer.token_to_id('[SOS]')
        self.EOS_ID = tokenizer.token_to_id('[EOS]')
        self.PAD_ID = tokenizer.token_to_id('[PAD]')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs, labels = self.data[idx].lower().split('_')
        inputs = tokenizer.encode(inputs.rstrip()).ids
        inputs = inputs + [self.PAD_ID] * (self.max_length - len(inputs))
        labels = tokenizer.encode(labels.replace('\n', '')).ids
        labels_shift = [self.SOS_ID] + labels
        labels = labels + [self.EOS_ID]

        if self.transform:
            inputs = self.transform(inputs)
            labels = self.transform(labels)
            labels_shift = self.transform(labels_shift)

        return inputs, labels_shift, labels


class DateDataModule(LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, num_workers=0):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor([int(i) for i in x], dtype=torch.long)),
        ])

        dataset = DateDataset(txt_file='data/date/date.txt', transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        return dataloader


tokenizer = Tokenizer.from_file("data/date/tokenizer.json")
vocab = tokenizer.get_vocab()


if __name__ == '__main__':
    datamodule = DateDataModule(batch_size=1)
    dataloader = datamodule.train_dataloader()

    for i, data in enumerate(dataloader):
        inputs, _, labels = data
        print(inputs.shape, labels.shape)
        print(inputs[0])
        print(labels[0])
        print(tokenizer.decode(inputs[0].tolist()).replace('   ', '[]').replace(' ', '').replace('[]', ' '))
        print(tokenizer.decode(labels[0].tolist()).replace('   ', '[]').replace(' ', '').replace('[]', ' '))
        break
