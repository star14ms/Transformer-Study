import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch import nn

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the tokenizer
tokenizer = get_tokenizer('spacy')

# Define the function to yield list of tokens
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Load the train dataset
train_iter = AG_NEWS(split='train')

# Build the vocab
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define the text and label processing functions
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# Define the collate function
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Create the data loader
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

# Define the model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Instantiate the model
num_class = len(set([label for (label, text) in train_iter]))
model = TextClassificationModel(len(vocab), 64, num_class).to(device)

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

# Train the model
for epoch in range(5):
    for i, (labels, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, labels)
       
