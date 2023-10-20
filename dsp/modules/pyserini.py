from typing import Union
import json
from datasets import Dataset

from dsp.utils import dotdict

try:
    from nltk.translate.chrf_score import sentence_chrf
except ImportError:
    raise ModuleNotFoundError(
        "You need to install nltk to use the remove_similar_passages function."
    )


class PyseriniRetriever:
    """Wrapper for retrieval with Pyserini. Supports using either pyserini prebuilt faiss indexes or your own faiss index."""

    def __init__(self, 
                 query_encoder: str = 'castorini/dkrr-dpr-nq-retriever', 
                 index: str = 'wikipedia-dpr-dkrr-nq', 
                 dataset: Dataset = None,
                 id_field: str = '_id',
                 text_fields: list[str] = ['text']) -> None:
        """
        Args:
        
            query_encoder (`str`):
                Huggingface model to encode queries
            index (`str`):
                Either a prebuilt index from pyserini or a local path to a faiss index
            dataset (`Dataset`):
                Only required when using a local faiss index. The dataset should be the one that has been put into the faiss index.
            id_field (`str`):
                The name of the id field of the dataset used for retrieval.
            text_fields (`list[str]`):
                A list of the names of the text fields for the dataset used for retrieval.
        """
        
        # Keep pyserini as an optional dependency
        from pyserini.search import FaissSearcher
        from pyserini.prebuilt_index_info import TF_INDEX_INFO, FAISS_INDEX_INFO, IMPACT_INDEX_INFO
        
        self.encoder = FaissSearcher._init_encoder_from_str(query_encoder)
        self.dataset = dataset
        self.id_field = id_field
        self.text_fields = text_fields
        
        if index in TF_INDEX_INFO or index in FAISS_INDEX_INFO or index in IMPACT_INDEX_INFO:
            self.searcher = FaissSearcher.from_prebuilt_index(index, self.encoder)
        else:
            self.searcher = FaissSearcher(index_dir=index, query_encoder=self.encoder)
            assert self.dataset is not None
            self.dataset_id_to_index = {}
            for i, docid in enumerate(self.dataset[self.id_field]):
                self.dataset_id_to_index[docid] = i
                

    def __call__(
        self, query: str, k: int = 10, threads: int = 16, remove_similar: bool = False
    ) -> Union[list[str], list[dotdict]]:
        # NOTE FV: we retrieve 2*k passages, then apply chrF filter (remove very
        # similar passages), hoping to get at least k passages (a bit dirty)

        hits = self.searcher.search(query, k=k*2, threads=threads)
        
        topk = []
        for rank, hit in enumerate(hits, start=1):
            if self.dataset is not None:
                row = self.dataset_id_to_index[hit.docid]
                text = ' | '.join(self.dataset[field][row] for field in self.text_fields)
                pid = self.dataset[self.id_field][row]
            else:
                # Pyserini prebuilt faiss indexes can perform docid lookup
                psg = json.loads(self.searcher.doc(hit.docid).raw())
                text = ' | '.join(psg[field] for field in self.text_fields)
                pid = psg[self.id_field]
            
            if remove_similar:
                text_wout_title = text.split(" | ")[1]
                current_texts = [psg['long_text'].split(" | ")[1] for psg in topk if 'long_text' in psg]
                is_similar = any(are_passages_similar(text_wout_title, txt) for txt in current_texts)
                if is_similar:
                    print("Removing a very similar passage")
                    continue

            topk.append({
                'text': text,
                'long_text': text,
                'pid': pid,
                'score': hit.score,
                'rank': rank,
            })

            if len(topk) == k:
                break
        
        return [dotdict(psg) for psg in topk]


def are_passages_similar(psg1: str, psg2: str, threshold: float = 0.95) -> bool:
    # NOTE chrF score works for any retriever
    # NOTE we dont use tf-idf because we need the idf of the whole corpus, not just the retrieved passages
    # TODO for pyserini, try to use return_vector arg from pyserini/search/faiss/_searcher.py#L442
    #   a) then, use cosine similarity to filter out similar passages
    #   b) would this work if we use the bm25 implementation of pyserini? or can we make
    #      it work with our BM25 implementation?
    score = sentence_chrf(psg1, psg2)
    return score > threshold
