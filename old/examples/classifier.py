from transformers import pipeline
from rich import print

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
x = classifier('We are very happy to introduce pipeline to the transformers repository.')
print(x)