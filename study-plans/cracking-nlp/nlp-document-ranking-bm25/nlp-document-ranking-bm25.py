from collections import Counter
import math
import numpy as np


def document_ranking(documents, query, k1=1.5, b=0.75, ndigits=4):
    """
    documents: list of token lists (pre-tokenized)
    query:     token list
    ndigits:   rounding applied to all emitted floats; pass None for full precision
    """
    N = len(documents)
    doc_tf = [Counter(d) for d in documents]
    doc_len = np.fromiter((len(d) for d in documents), dtype=np.float64, count=N)
    avgdl = doc_len.mean() if N else 0.0

    def rnd(x):
        return x if ndigits is None else round(x, ndigits)

    idf_values, term_contributions = {}, {}
    scores = np.zeros(N, dtype=np.float64)

    if N and avgdl > 0.0:
        norm = k1 * (1.0 - b + b * doc_len / avgdl)
        for q, qtf in Counter(query).items():
            df = sum(1 for tf in doc_tf if q in tf)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            f = np.fromiter((tf.get(q, 0) for tf in doc_tf), dtype=np.float64, count=N)
            contrib = qtf * idf * np.divide(
                f * (k1 + 1.0), f + norm, out=np.zeros(N), where=f > 0
            )
            scores += contrib
            idf_values[q] = idf
            term_contributions[q] = [c for c in contrib.tolist()]
    else:
        for q in Counter(query):
            idf_values[q] = rnd(0.0)
            term_contributions[q] = [rnd(0.0)] * N

    return {
        "avg_doc_length": rnd(avgdl),
        "idf_values": idf_values,
        "ranking": np.argsort(-scores, kind="stable").tolist(),
        "scores": [rnd(s) for s in scores.tolist()],
        "term_contributions": term_contributions,
    }