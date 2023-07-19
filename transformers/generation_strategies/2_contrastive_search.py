from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

checkpoint = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

prompt = "Hugging Face Company is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
