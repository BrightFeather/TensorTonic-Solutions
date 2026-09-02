import math

def tfidf(corpus):
    """
    Returns: dict
    """
    words = list(set([w for d in corpus for w in d]))
    words.sort()
    N = len(corpus)
    res = [[0 for _ in range(len(words))] for _ in range(len(corpus))]
    
    for i, w in enumerate(words):
        df = sum([1 if w in d else 0 for d in corpus])
        idf = math.log(N*1.0/df)
        for j, d in enumerate(corpus):
            tf = d.count(w)
            res[j][i] = round(tf * idf, 4)
    return {
        'vocab':dict(zip(words, range(len(words)))),
        'tfidf':res
    }
        
        
    