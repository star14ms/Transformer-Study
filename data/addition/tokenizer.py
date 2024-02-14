from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, Whitespace
from tokenizers.trainers import BpeTrainer
import json


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Use a pre-tokenizer that splits everything into single characters.
# The Split pre-tokenizer with pattern "" and behavior "isolated" can achieve this.
tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

# make exception that doesn't tokenize
tokenizer.add_special_tokens(["\n", " "])

trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
with open("data/addition/addition.txt", "r") as f:
    data = (line.replace("_", "") for line in f.readlines())
tokenizer.train_from_iterator(data, trainer)

output = tokenizer.encode("16+75 _91")
print("Tokens:", output.tokens)
print("IDs:", output.ids)

# Decode back to text
decoded_text = tokenizer.decode(output.ids)
print("Decoded Text:", decoded_text)

vocab = tokenizer.get_vocab()
# sort vocabs by id
vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
print('Vocabulary:', vocab)
print('Vocabulary size:', len(vocab))

# save the vocab
with open("data/addition/vocab.json", "w") as f:
    json.dump(vocab, f, indent=4)

# save the tokenizer
tokenizer.save("data/addition/tokenizer.json")
tokenizer = Tokenizer.from_file("data/addition/tokenizer.json")
