from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

prompt = "The Permaculture Design Principles are a set of universal design principles... \
  that can be applied to any location, climate and culture, and they allow us to design \
  the most efficient and sustainable human habitation and food production systems. \
  Permaculture is a design system that encompasses a wide variety of disciplines, such \
  as ecology, landscape design, environmental science and energy conservation, and the \
  Permaculture design principles are drawn from these various disciplines. Each individual \
  design principle itself embodies a complete conceptual framework based on sound \
  scientific principles. When we bring all these separate  principles together, we can \
  create a design system that both looks at whole systems, the parts that these systems \
  consist of, and how those parts interact with each other to create a complex, dynamic, \
  living system. Each design principle serves as a tool that allows us to integrate all \
  the separate parts of a design, referred to as elements, into a functional, synergistic, \
  whole system, where the elements harmoniously interact and work together in the most \
  efficient way possible."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, num_beams=5, num_beam_groups=5, max_new_tokens=30, diversity_penalty=1.0)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
