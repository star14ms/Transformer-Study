import torch
from data.addition import tokenizer, AdditionDataModule
from model_lighting import AdditionTransformerEncoderL, AdditionTransformerL # choose the model you want to test
from device import get_device
from thop import profile
from rich import print as pprint


def test():
    for data in dataloader:
        decode = lambda x: tokenizer.decode(x.tolist()).replace(' ', '')

        if label_type == 'string':
            inputs, _, labels = data
            inputs = inputs.to(device)
            outputs = torch.tensor([SOS_ID] * BATCH_SIZE * 1).view(BATCH_SIZE, 1).to(device)

            while True:
                probs = model(inputs, outputs)
                targets = probs.argmax(dim=2)[:, -1].view(BATCH_SIZE, 1)
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
        pprint(f'[green]{str(label).ljust(5)}[/green] <- prediction')
        pprint(f'[yellow]{str(output).ljust(5)}[/yellow] <- output')
        input()


if __name__ == '__main__':
    MAX_LENGTH = 5
    BATCH_SIZE = 1
    SOS_ID = tokenizer.token_to_id('[SOS]')
    EOS_ID = tokenizer.token_to_id('[EOS]')

    model = AdditionTransformerEncoderL() # change the model

    label_type = 'int' if model.__class__.__name__ == 'AdditionTransformerEncoderL' else 'string'
    datamodule = AdditionDataModule(batch_size=BATCH_SIZE, label_type=label_type)
    dataloader = datamodule.train_dataloader()

    model.load_state_dict(torch.load('models/model_encoder.pth'))
    device = get_device()
    model.to(device)

    macs, params = profile(model, inputs=(
        torch.ones(1, 7, dtype=torch.int).to(device), 
    ) + ((
        torch.zeros(1, 5, dtype=torch.int).to(device),
    ) if label_type == 'string' else ()), verbose=False)
    print('\n모델 생성 완료! (MACs: {:.0f} | Params: {:.0f})\n'.format(macs, params))

    test()