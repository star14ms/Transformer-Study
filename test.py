import torch
from data.addition import tokenizer, vocab, AdditionDataModule
from model_lighting import TransformerEncoderL, TransformerL
from device import get_device
from thop import profile
from rich import print as pprint


def test():
    for (inputs, _, labels) in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.tensor([SOS_ID] * batch_size * 1).view(batch_size, 1).to(device)
        decode = lambda x: tokenizer.decode(x.tolist()).replace(' ', '')

        while True:
            probs = model(inputs, outputs)
            targets = probs.argmax(dim=2)[:, -1].view(batch_size, 1)
            outputs = torch.cat([outputs, targets], dim=1)
            if outputs[0, -1].item() == EOS_ID or outputs.shape[1] == MAX_LENGTH:
                break

        print(decode(inputs[0]))
        label = decode(labels[0])
        output = decode(outputs[0])
        # label = labels[0].item()
        # output = round(outputs[0].item(), 1)
        pprint(f'[green]{label}[/green]')
        pprint(f'[yellow]{output}[/yellow]')
        # pprint(torch.round(probs[0][:].softmax(dim=1).to('cpu'), decimals=1).tolist())
        # print(f'{abs(label - output):.1f} < difference', )
        input()


if __name__ == '__main__':
    batch_size = 1
    datamodule = AdditionDataModule(batch_size=batch_size, label_type='string')
    dataloader = datamodule.train_dataloader()

    SOS_ID = tokenizer.token_to_id('[SOS]')
    EOS_ID = tokenizer.token_to_id('[EOS]')

    # model = TransformerEncoderL(
    #     vocab_size=len(vocab),
    #     d_model=32,
    #     nhead=8,
    #     num_layers=1,
    #     dim_feedforward=32,
    #     batch_first=True,
    #     max_length=4,
    # )
    model = TransformerL(
        vocab_size=len(vocab), 
        d_model=32, 
        nhead=8, 
        num_encoder_layers=1, 
        num_decoder_layers=1, 
        dim_feedforward=32, 
        PAD_ID=tokenizer.token_to_id('[PAD]'),
        batch_first=True
    )
    model.load_state_dict(torch.load('models/model.pth'))
    device = get_device()
    model.to(device)

    MAX_LENGTH = 5
    macs, params = profile(model, inputs=(
        torch.ones(1, 7, dtype=torch.int).to(device), 
        torch.zeros(1, 5, dtype=torch.int).to(device)
    ), verbose=False)
    print('\n모델 생성 완료! (MACs: {:.0f} | Params: {:.0f})\n'.format(macs, params))

    test()