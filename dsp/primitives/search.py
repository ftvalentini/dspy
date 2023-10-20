import numpy as np
import dsp


def retrieve(query: str, k: int, **kwargs) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=k, **kwargs)
    # NOTE remove_similar is implemented in each retriever e.g. Pyserini
    passages = [psg.long_text for psg in passages]
    
    if dsp.settings.reranker:
        passages_cs_scores = dsp.settings.reranker(query, passages)
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
        passages = [passages[idx] for idx in passages_cs_scores_sorted]

    return passages


def retrieveRerankEnsemble(queries: list[str], k: int) -> list[str]:
    if not (dsp.settings.rm and dsp.settings.reranker):
        raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
    queries = [q for q in queries if q]
    passages = {}
    for query in queries:
        retrieved_passages = dsp.settings.rm(query, k=k*3)
        passages_cs_scores = dsp.settings.reranker(query, [psg.long_text for psg in retrieved_passages])
        for idx in np.argsort(passages_cs_scores)[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = passages.get(psg.long_text, []) + [
                passages_cs_scores[idx]
            ]

    passages = [(np.average(score), text) for text, score in passages.items()]
    return [text for _, text in sorted(passages, reverse=True)[:k]]


def retrieveEnsemble(queries: list[str], k: int, by_prob: bool = True, remove_similar: bool = False) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if dsp.settings.reranker:
        return retrieveRerankEnsemble(queries, k)
    
    queries = [q for q in queries if q]

    if len(queries) == 1:
        return retrieve(queries[0], k, remove_similar=remove_similar)

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, k=k * 3):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages


# FV NOT USED:
# 
# # To measure similarity, we drop title from passages (psg.split(" | ")[1])
# passages = dsp.settings.rm(query, k=k*2, **kwargs)
# passages = [psg.long_text for psg in passages]
# passages = remove_similar_passages(passages)
# passages = passages[:k]
# 
# def remove_similar_passages(passages):
#     # Remove title from passages and add an index to keep track of the original order
#     texts = [psg.split(" | ")[1] for psg in passages]
#     # Initialize with first passage:
#     unique_texts = [texts[0]]
#     indices = [0]
#     # Iterate through the sorted passages
#     for i in range(1, len(texts)):
#         current_text = texts[i]
#         is_similar = any(are_passages_similar(current_text, txt) for txt in unique_texts)
#         # If not similar, add to the list of unique passages
#         if not is_similar:
#             unique_texts.append(current_text)
#             indices.append(i)
#     filtered_passages = [passages[idx] for idx in indices]
#     return filtered_passages

