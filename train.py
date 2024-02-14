from torch import nn
from torch.nn.functional import one_hot
import torch
from data.addition import tokenizer, vocab, AdditionDataModule
from model import Transformer
import math
from device import get_device
from thop import profile


model = Transformer(vocab_size=len(vocab), d_model=32, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, batch_first=True)
device = get_device()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

datamodule = AdditionDataModule(batch_size=128)
dataloader = datamodule.train_dataloader(label_type='string')
PAD_ID = tokenizer.token_to_id('[PAD]')
max_epoch = 10

# macs, params = profile(model, inputs=(
#   torch.ones(1, 7, dtype=torch.int).to(device), 
#   torch.zeros(1, 5, dtype=torch.int).to(device))
# , verbose=False)
# print('\n모델 생성 완료! (MACs: {:.0f} | Params: {:.0f})\n'.format(macs, params))

def train_epoch():
    max_iter = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
    loss_sum = 0

    for i, data in enumerate(dataloader):
        inputs, labels_shift, labels = data
        inputs, labels_shift, labels = inputs.to(device), labels_shift.to(device), labels.to(device)

        kwargs = {
            # 'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            # 'memory_mask': memory_mask,
            'src_key_padding_mask': (inputs == PAD_ID).to(torch.float32),
            'tgt_key_padding_mask': (labels == PAD_ID).to(torch.float32),
            # 'memory_key_padding_mask': (inputs == PAD_ID).to(torch.float32),
            # 'tgt_is_causal': True,
            # 'src_is_causal': True,
            # 'memory_is_causal': True,
        }
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, labels_shift, **kwargs)
        outputs_prob = torch.softmax(outputs, dim=2) # (N, L, V) (N: batch size, L: tgt length, V: vocab size)
        labels = one_hot(labels, num_classes=len(vocab)).to(torch.float32)
        loss = criterion(outputs_prob, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if i % 100 == 0:
            print(f'Iter: {i}/{max_iter} | Loss: {loss.item():.6f}')

    avg_loss = loss_sum / max_iter

    return avg_loss


src_mask = nn.Transformer.generate_square_subsequent_mask(7).to(device)
# memory_mask = nn.Transformer.generate_square_subsequent_mask(7).to(device)
tgt_mask = nn.Transformer.generate_square_subsequent_mask(5).to(device)


# train code
for epoch in range(max_epoch):
    loss_avg = train_epoch()

    print(f'Epoch: {epoch+1} | Loss: {loss_avg:.6f}')

    if loss_avg < 0.001:
        break


# save model
if max_epoch > 0:
    torch.save(model.state_dict(), 'model.pth')
    print('Finished Training')
