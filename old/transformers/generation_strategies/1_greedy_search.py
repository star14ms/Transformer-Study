from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

prompt = "I look forward to"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs) # /transformers/generation/utils.py(1149)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
