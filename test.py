import nltk
from src.utils._tokenizer import tokenize

s = "The quick brown fox jumps over the lazy dog."
tokens = tokenize([s], word_process_method="stem", return_ids=False)

print(tokens)
tokens = nltk.word_tokenize(s)
print(tokens)
