from collections import Counter
from src import schemas
from src.core import config
from src.utils import tokenize


def build_inverted_index(
    doc_ids: list[str],
    texts: list[str],
    word_process_method: str = config.WORD_PROCESS_METHOD,
) -> tuple[dict[str, schemas.TermEntry], dict[str, int]]:
    """Returns postings list and document lengths for the given texts."""

    if len(doc_ids) != len(texts):
        raise ValueError("doc_ids and texts must have the same length")
    inverted_index: dict[str, schemas.TermEntry] = {}
    doc_lens: dict[str, int] = {}

    tokenized_docs = tokenize(texts=texts, word_process_method=word_process_method)

    # creating postings list
    for doc_id, tokens in zip(doc_ids, tokenized_docs):
        doc_lens[doc_id] = len(tokens)
        term_counts = Counter(tokens)

        for token, term_freq in term_counts.items():
            if token not in inverted_index:
                inverted_index[token] = schemas.TermEntry(doc_freq=0, postings=[])

            inverted_index[token].postings.append(
                schemas.PostingEntry(doc_id=doc_id, term_freq=term_freq)
            )

    # compute document frequencies
    for term, term_entry in inverted_index.items():
        inverted_index[term].doc_freq = len(term_entry.postings)

    return inverted_index, doc_lens
