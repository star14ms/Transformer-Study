import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from rich import print

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

while True:
    text = input('input text: ')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    
    _input = tokenizer.decode(inputs["input_ids"][0])
    _output = tokenizer.decode(torch.argmax(logits, dim=2)[0])
    
    print(_input)
    print(_output)
    print('-' * 80)
    