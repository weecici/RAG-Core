import ir_datasets
from llama_index.core.schema import TextNode


def load_ir_dataset(dataset_name: str) -> list[TextNode]:
    ds = ir_datasets.load(dataset_name)
    return []
