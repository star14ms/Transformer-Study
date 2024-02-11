from torch import nn
from torch.nn.functional import one_hot
import torch
from data.addition import tokenizer, vocab, AdditionDataModule
from model import TransformerEncoder, Transformer
import math
from device import get_device
from thop import profile


model = Transformer(vocab_size=len(vocab), d_model=32, nhead=8, num_encoder_layers=1, num_decoder_layers=3, dim_feedforward=32, batch_first=True)
device = get_device()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

datamodule = AdditionDataModule(batch_size=128)
dataloader = datamodule.train_dataloader(label_type='string')
batch_size = dataloader.batch_size
max_iter = math.ceil(len(dataloader.dataset) / batch_size)
MAX_LENGTH = 4
TGT = torch.tensor(tokenizer.encode('    ').ids * batch_size).view(batch_size, MAX_LENGTH).to(device)
max_epoch = 20

macs, params = profile(model, inputs=(torch.ones(1, 7, dtype=torch.int).to('mps'), TGT[0, :]), verbose=False)
print('\n모델 생성 완료! (MACs: {:.0f} | Params: {:.0f})\n'.format(macs, params))

def train_epoch():
    loss_sum = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = model(inputs, TGT[:labels.shape[0], :])
        labels = one_hot(labels, num_classes=len(vocab)).to(torch.float32)
        outputs_prob = torch.softmax(outputs, dim=2) # (N, L, V) (N: batch size, L: tgt length, V: vocab size)
        loss = criterion(outputs_prob, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if i % 100 == 0:
            print(f'Iter: {i}/{max_iter} | Loss: {loss.item():.6f}')

    avg_loss = loss_sum / max_iter

    return avg_loss


# train code
for epoch in range(max_epoch):
    loss_avg = train_epoch(epoch)

    print(f'Epoch: {epoch+1} | Loss: {loss_avg:.6f}')

    if loss_avg < 0.001:
        break


# save model
if max_epoch > 0:
    torch.save(model.state_dict(), 'model.pth')
    print('Finished Training')
