import numpy as np

def minhash_dedup(documents, num_hashes, shingle_size, threshold, seed=42):
    """
    Returns: dict
    """
    P = (1 << 31) - 1  # Mersenne prime

    # Step 1: Shingling
    all_shingles = []
    for doc in documents:
        shingles = set()
        for i in range(len(doc) - shingle_size + 1):
            shingles.add(doc[i:i + shingle_size])
        all_shingles.append(sorted(list(shingles)))

    # Step 2: Shingle to integer
    def shingle_to_int(s):
        val = 0
        for c in s:
            val = (val * 256 + ord(c)) % P
        return val

    shingle_ints = []
    for shingles in all_shingles:
        shingle_ints.append([shingle_to_int(s) for s in shingles])

    # Step 3: Hash function coefficients
    rng = np.random.RandomState(seed)
    a_coeffs = rng.randint(1, P, size=num_hashes).tolist()
    b_coeffs = rng.randint(0, P, size=num_hashes).tolist()

    # Step 4: MinHash signatures
    N = len(documents)
    signatures = []
    for doc_idx in range(N):
        sig = []
        ints = shingle_ints[doc_idx]
        for k in range(num_hashes):
            if len(ints) == 0:
                sig.append(P)
            else:
                min_hash = P
                for x in ints:
                    h = (a_coeffs[k] * x + b_coeffs[k]) % P
                    if h < min_hash:
                        min_hash = h
                sig.append(min_hash)
        signatures.append(sig)

    # Step 5: Pairwise estimated Jaccard
    estimated_similarities = []
    for i in range(N):
        for j in range(i + 1, N):
            matching = sum(1 for k in range(num_hashes)
                          if signatures[i][k] == signatures[j][k])
            sim = round(matching / num_hashes, 4)
            estimated_similarities.append([i, j, sim])

    # Step 6: Greedy deduplication
    duplicate_pairs = []
    is_duplicate = [False] * N
    for i in range(N):
        if is_duplicate[i]:
            continue
        for j in range(i + 1, N):
            if is_duplicate[j]:
                continue
            sim = None
            for entry in estimated_similarities:
                if entry[0] == i and entry[1] == j:
                    sim = entry[2]
                    break
            if sim is not None and sim >= threshold:
                duplicate_pairs.append([i, j])
                is_duplicate[j] = True

    unique_indices = [i for i in range(N) if not is_duplicate[i]]

    return {
        'shingles': all_shingles,
        'signatures': signatures,
        'estimated_similarities': estimated_similarities,
        'duplicate_pairs': duplicate_pairs,
        'unique_indices': unique_indices
    }
