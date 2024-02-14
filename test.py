import torch
from data.addition import tokenizer, vocab, AdditionDataModule
from model_lighting import AdditionTransformerEncoderL, AdditionTransformerL
from device import get_device
from thop import profile
from rich import print as pprint


def test():
    for data in dataloader:
        decode = lambda x: tokenizer.decode(x.tolist()).replace(' ', '')

        if label_type == 'string':
            inputs, _, labels = data
            inputs = inputs.to(device)
            outputs = torch.tensor([SOS_ID] * batch_size * 1).view(batch_size, 1).to(device)

            while True:
                probs = model(inputs, outputs)
                targets = probs.argmax(dim=2)[:, -1].view(batch_size, 1)
                outputs = torch.cat([outputs, targets], dim=1)
                if outputs[0, -1].item() == EOS_ID or outputs.shape[1] == MAX_LENGTH:
                    break

            label = decode(labels[0])
            output = decode(outputs[0])
        else:
            inputs, labels = data
            outputs = model(inputs.to(device))

            label = labels[0].item()
            output = round(outputs[0].item(), 1)
        
        print(decode(inputs[0]))
        pprint(f'[green]{label}[/green]')
        pprint(f'[yellow]{output}[/yellow]')

        input()


if __name__ == '__main__':
    MAX_LENGTH = 5
    batch_size = 1
    label_type = 'string'
    datamodule = AdditionDataModule(batch_size=batch_size, label_type=label_type)
    dataloader = datamodule.train_dataloader()

    SOS_ID = tokenizer.token_to_id('[SOS]')
    EOS_ID = tokenizer.token_to_id('[EOS]')

    model = AdditionTransformerL()
    model.load_state_dict(torch.load('models/model.pth'))
    device = get_device()
    model.to(device)

    # macs, params = profile(model, inputs=(
    #     torch.ones(1, 7, dtype=torch.int).to(device), 
    #     torch.zeros(1, 5, dtype=torch.int).to(device)
    # ), verbose=False)
    # print('\n모델 생성 완료! (MACs: {:.0f} | Params: {:.0f})\n'.format(macs, params))

    test()