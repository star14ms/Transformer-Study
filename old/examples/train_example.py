import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim

# Set up fields
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

# Suppose we're doing sentiment analysis, and we have a split of training/testing data
# in the form of tsv files (train.tsv, test.tsv)
train_data, test_data = data.TabularDataset.splits(
    path='your/directory', train='train.tsv', test='test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])

# Build the vocab
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

# Choose a batch size
batch_size = 64

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the iterators
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=device)

breakpoint()

# Define the LSTM model
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.rnn(embedded)
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))

# Create an instance of the model
model = RNN(len(TEXT.vocab), 100, 256, 1)

# Choose an optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Train the model
for epoch in range(5):  # 5 epochs, you may need more
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.Text).squeeze(1)
        loss = criterion(predictions, batch.Label)
        loss.backward()
        optimizer.step()

# Save the model for later use
torch.save(model.state_dict(), 'model.pt')
