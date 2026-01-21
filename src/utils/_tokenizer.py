import nltk
import string
from nltk import WordNetLemmatizer, SnowballStemmer, word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from typing import Literal

_stopwords = set(stopwords.words("english"))
_punctuations: set[str] = set(string.punctuation)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # Default


stemmer = SnowballStemmer("english")
lmtz = WordNetLemmatizer()


def tokenize(
    texts: list[str],
    word_process_method: Literal["lemmatize", "stem"] = "lemmatize",
    verbose: bool = False,
) -> list[list[str]]:

    # 1. Tokenize RAW text (Keep Case for POS Tagging accuracy)
    tok_lists = [word_tokenize(text) for text in texts]

    if verbose:
        raw_set = set()
        for tokens in tok_lists:
            raw_set.update([t.lower() for t in tokens])

    final_tok_lists = []

    if word_process_method == "lemmatize":
        for tokens in tok_lists:
            # STEP A: Tag the RAW tokens
            tagged = pos_tag(tokens)

            processed_tokens = []
            for word, tag in tagged:
                # STEP B: Lowercase ONLY when processing/checking
                word_lower = word.lower()

                if word_lower not in _stopwords and word_lower not in _punctuations:
                    # Filter out purely punctuation tokens (like "--") but keep text
                    if not all(char in _punctuations for char in word_lower):
                        wn_tag = get_wordnet_pos(tag)
                        processed_tokens.append(lmtz.lemmatize(word_lower, wn_tag))
            final_tok_lists.append(processed_tokens)

    else:  # Stemming
        for tokens in tok_lists:
            # Fix: Lowercase first, then filter, then stem INDIVIDUAL words
            filtered = [
                t.lower()
                for t in tokens
                if t.lower() not in _stopwords and t.lower() not in _punctuations
            ]
            # FIX: Iterate correctly
            stemmed = [stemmer.stem(w) for w in filtered]
            final_tok_lists.append(stemmed)

    if verbose:
        final_set = set()
        for tokens in final_tok_lists:
            final_set.update(tokens)
        print(f"Original vocabulary size: {len(raw_set)}")
        print(f"Final vocabulary size: {len(final_set)}")
        print(f"Percent of vocabulary retained: {len(final_set)/len(raw_set)*100:.2f}%")
        print(f"Example terms: {list(final_set)[:10]}")

    return final_tok_lists
