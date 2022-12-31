from transformers import pipeline
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello", return_tensors="pt")
print(inputs)
print("b")
print("c")
print("d")
# how to train word embdings: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
# https://d2l.ai/chapter_introduction/index.html
# https://www.youtube.com/watch?v=FKlPCK1uFrc&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6
# https://www.youtube.com/watch?v=FKlPCK1uFrc&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6https://www.youtube.com/watch?v=FKlPCK1uFrc&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6
# https://d2l.ai/chapter_introduction/index.html
# https://www.youtube.com/watch?v=NgWujOrCZFo&list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK

import FTE_NLP.utils.domainadaption_data_clean