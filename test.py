import torch
from data.addition import tokenizer as tokenizer_addition, AdditionDataModule
from data.date import tokenizer as tokenizer_date, DateDataModule
from model_lighting import AdditionTransformerEncoderL, AdditionTransformerL, DateTransformerL # choose the model you want to test
from device import get_device
from thop import profile
from rich import print as pprint
from visualize import show_attn_weight


def test():
    for data in dataloader:
        decode = lambda x: tokenizer.decode(x.tolist()).replace('   ', '[]').replace(' ', '').replace('[]', ' ')

        if label_type == 'string':
            inputs, _, labels = data
            inputs = inputs.to(device)
            outputs = torch.tensor([SOS_ID] * BATCH_SIZE * 1).view(BATCH_SIZE, 1).to(device)

            while True:
                probs = model(inputs, outputs)
                # print top 5 values 
                x = probs.softmax(dim=2).topk(5, dim=2)
                print(tokenizer.decode(x.indices[0, -1].tolist()))
                # print(x.indices[0, -1].tolist())
                targets = probs.argmax(dim=2)[:, -1].view(BATCH_SIZE, 1)
                outputs = torch.cat([outputs, targets], dim=1)
                if outputs[0, -1].item() == EOS_ID:
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
        # show_attn_weight(inputs, model, decode(inputs[0]).ljust(len(inputs[0])), output)
        input()


if __name__ == '__main__':
    BATCH_SIZE = 1
    
    model = DateTransformerL() # change the model
    label_type = 'int' if model.__class__.__name__ == 'AdditionTransformerEncoderL' else 'string'

    if 'addition' in model.__class__.__name__.lower():
        tokenizer = tokenizer_addition
        datamodule = AdditionDataModule(batch_size=BATCH_SIZE, label_type=label_type)
    elif 'date' in model.__class__.__name__.lower():
        tokenizer = tokenizer_date
        datamodule = DateDataModule(batch_size=BATCH_SIZE)
    dataloader = datamodule.train_dataloader()

    SOS_ID = tokenizer.token_to_id('[SOS]')
    EOS_ID = tokenizer.token_to_id('[EOS]')

    model.load_state_dict(torch.load('models/model_date.pth'))
    device = get_device()
    model.to(device)
    model.eval()

    # macs, params = profile(model, inputs=(
    #     torch.ones(1, 7, dtype=torch.int).to(device), 
    # ) + ((
    #     torch.zeros(1, 5, dtype=torch.int).to(device),
    # ) if label_type == 'string' else ()), verbose=False)
    # print('\n모델 생성 완료! (MACs: {:.0f} | Params: {:.0f})\n'.format(macs, params))


    with torch.no_grad():
        test()