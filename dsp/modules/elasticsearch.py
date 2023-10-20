
from typing import Optional, Union, Any

import numpy as np
from dsp.utils import dotdict

try:
    from nltk.translate.chrf_score import sentence_chrf
except ImportError:
    raise ModuleNotFoundError(
        "You need to install nltk to use the remove_similar_passages function."
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
        self, query: str, k: int = 10, remove_similar: bool = False
    ) -> Union[list[str], list[dotdict]]:
        # NOTE FV: we retrieve 2*k passages, then apply chrF filter (remove very
        # similar passages), hoping to get at least k passages (a bit dirty)
        hits: list[dict[str, Any]] = elastic_search_request(
            self.retriever, query, k=k*2)
        # NOTE this just tries to match the format of the other retrievers
        topk = [{**hits[0], "long_text": hits[0]["text"]}]
        for hit in hits[1:]:
            if remove_similar:
                current_texts = [psg['long_text'].split(" | ")[1] for psg in topk]
                text_wout_title = hit["text"].split(" | ")[1]
                is_similar = any(are_passages_similar(text_wout_title, txt) for txt in current_texts)
                if is_similar:
                    print("Removing a very similar passage")
                    continue
            topk.append({**hit, "long_text": hit["text"]})
            if len(topk) == k:
                break
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


def are_passages_similar(psg1: str, psg2: str, threshold: float = 0.95) -> bool:
    # NOTE chrF score works for any retriever
    # NOTE we dont use tf-idf because we need the idf of the whole corpus, not just the retrieved passages
    score = sentence_chrf(psg1, psg2)
    return score > threshold
