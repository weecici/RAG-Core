import ir_datasets
from typing import Literal
from ir_datasets.datasets.cranfield import CranfieldDoc
from llama_index.core.schema import TextNode


def load_ir_dataset(dataset_name: Literal["cranfield"]) -> list[TextNode]:
    docs = ir_datasets.load(dataset_name)
    nodes: list[TextNode] = []
    if dataset_name == "cranfield":
        for doc in docs.docs_iter():
            doc: CranfieldDoc

            text = doc.text.replace("\n", " ").strip()
            author = doc.author.replace("\n", " ").strip()
            bib = doc.bib.replace("\n", " ").strip()
            text = f"{text}\nAuthor: {author}\nBibliography: {bib}"
            title = doc.title.replace("\n", " ").strip()
            node = TextNode(
                id_=str(doc.doc_id),
                text=text,
                metadata={"title": title, "file_path": ""},
            )
            nodes.append(node)
        return nodes
    return nodes
