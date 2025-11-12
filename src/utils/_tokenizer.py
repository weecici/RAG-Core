import nltk
import string
from nltk import WordNetLemmatizer, SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from typing import Literal


nltk.download("wordnet")
_stopwords = set(stopwords.words("english"))
_punctuations: set[str] = set([p for p in string.punctuation])


def stem(words: list[str]) -> list[str]:
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in words]


def lemmatize(words: list[str]) -> list[str]:
    lmtz = WordNetLemmatizer()
    return [lmtz.lemmatize(word) for word in words]


def tokenize(
    texts: list[str],
    word_process_method: Literal["lemmatize", "stem"] = "stem",
) -> list[list[str]]:
    process_method: callable[[list[str]], list[str]] = stem
    if word_process_method == "lemmatize":
        process_method = lemmatize

    tok_lists = [word_tokenize(text) for text in texts]

    sw_punc_rm_tok_lists = [
        [
            token
            for token in tokens
            if not (token.lower() in _stopwords or token in _punctuations)
        ]
        for tokens in tok_lists
    ]

    final_tok_lists = [process_method(tokens) for tokens in sw_punc_rm_tok_lists]
    return final_tok_lists
