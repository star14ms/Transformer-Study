from torch import nn
from torch.nn.functional import one_hot
import torch
from data.addition import tokenizer, vocab, AdditionDataModule
from model_lighting import TransformerEncoderL
from device import get_device
from thop import profile

datamodule = AdditionDataModule(batch_size=1)
dataloader = datamodule.train_dataloader(label_type='int', num_workers=0)

model = TransformerEncoderL(
    vocab_size=len(vocab),
    d_model=32,
    nhead=8,
    num_layers=1,
    dim_feedforward=32,
    batch_first=True,
    max_length=4,
)
device = get_device()
model.to(device)


def test():
    # load model
    model.load_state_dict(torch.load('model0.pth'))

    # test code
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        print(tokenizer.decode(inputs[0].tolist()).replace(' ', ''))
        label = labels[0].item()
        output = round(outputs[0].item(), 1)
        print(label, '< label')
        print(output, '< output')
        print(f'{abs(label - output):.1f} < difference', )
        input()


if __name__ == '__main__':
    test()