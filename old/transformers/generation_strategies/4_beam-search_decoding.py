from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

checkpoint = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

prompt = "It is astonishing how one can"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
