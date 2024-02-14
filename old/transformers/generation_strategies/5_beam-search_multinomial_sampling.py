from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5 import T5ForConditionalGeneration

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

prompt = "translate English to German: The house is wonderful."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, num_beams=5, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
