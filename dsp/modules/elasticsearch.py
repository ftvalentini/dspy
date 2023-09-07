
from typing import Optional, Union, Any

from dsp.utils import dotdict

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "You need to install numpy library"
        "Please use the command: pip install numpy"
    )

# TODO ideally we should import from elasticsearch lib instead of FLARE
# This will work only in my project, not in yours!
import sys
sys.path.append("scripts")
from FLARE.src.retriever import BM25


class ElasticSearch:
    """Wrapper for the ElasticSearch BM25 Retrieval."""

    def __init__(
        self,
        index_name: str,
        url: str = "http://localhost:9200",
        port: Optional[Union[str, int]] = None,
    ):
        self.index_name = index_name
        self.url = f"{url}:{port}" if port else url
        # load index
        self.retriever = BM25(index_name=self.index_name, engine="elasticsearch")

    def __call__(
        self, query: str, k: int = 10
    ) -> Union[list[str], list[dotdict]]:
        topk: list[dict[str, Any]] = elastic_search_request(
            self.retriever, query, k)
        # NOTE the folllowing just tries to match the format of the other 
        # retrievers
        topk = [{**d, "long_text": d["text"]} for d in topk]
        return [dotdict(psg) for psg in topk]


def elastic_search_request(retriever: BM25, query: str, k: int):
    """Search in ElasticSearch index
    """
    query = [query] # FLARE/Beir expects a list
    ids, titles, texts = retriever.retrieve(
        queries=query, topk=k, max_query_length=None)
    results = format_elastic_result(ids, titles, texts)
    return results[:k]


def format_elastic_result(
        ids: np.ndarray, titles: np.ndarray, texts: np.ndarray) -> list[dict]:
    """Format ElasticSearch result to format expected by dspy framework
    NOTE we add IDs just in case
    NOTE we already formatted chunks as "title | text" when creating index
    """
    ids = ids.flatten().tolist()
    titles = titles.flatten().tolist()
    texts = texts.flatten().tolist()
    results = []
    for id, title, text in zip(ids, titles, texts):
        results.append({"id": id, "title": title, "text": text})
    return results
