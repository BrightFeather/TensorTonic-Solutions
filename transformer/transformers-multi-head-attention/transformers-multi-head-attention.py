import numpy as np

def softmax(x, axis=-1):
    # subtract row max: mathematically a no-op (shift cancels in the ratio),
    # but prevents exp() overflowing to inf. largest exponent becomes exactly 0.
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    batch_size, seq_len, d_model = Q.shape        # (2, 5, 12)
    d_k = d_model // num_heads                    # 12 // 4 = 3

    # ---- 1. project ------------------------------------------------------
    # (B, L, d_model) @ (d_model, d_model) -> (B, L, d_model)
    # one big GEMM per input; the h heads are already independent columns
    # inside W, so slicing them out afterwards is equivalent to h small GEMMs.
    Q = np.dot(Q, W_q)                            # (2, 5, 12)
    K = np.dot(K, W_k)                            # (2, 5, 12)
    V = np.dot(V, W_v)                            # (2, 5, 12)

    # ---- 2. split the feature axis into heads ----------------------------
    # reshape:   (B, L, d_model) -> (B, L, h, d_k)      (2,5,12) -> (2,5,4,3)
    #            free: contiguous, just reinterprets the last axis
    # transpose: (B, L, h, d_k) -> (B, h, L, d_k)       (2,5,4,3) -> (2,4,5,3)
    #            free: a view, only strides change (result is NON-contiguous)
    #
    # the transpose is the load-bearing step. matmul contracts the last TWO
    # axes and treats everything before as batch. we need (L, d_k) as the
    # matrix and h as a batch dim, so h must move left of L. skip it and you
    # get (h,h) scores per position -- tokens never see each other -- with
    # no error raised.
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    #                                              all three now (2, 4, 5, 3)

    # ---- 3. scaled dot-product attention ---------------------------------
    # K.transpose(0,1,3,2): (B,h,L,d_k) -> (B,h,d_k,L)     (2,4,5,3)->(2,4,3,5)
    #   puts d_k on the inside so the contraction is over features, leaving
    #   both sequence axes as rows and columns of the score matrix.
    # matmul: (B,h,L_q,d_k) @ (B,h,d_k,L_k) -> (B,h,L_q,L_k)
    #   scores[b,i,s,t] = <q_s, k_t> for head i.
    # /sqrt(d_k): dot of d_k unit-variance terms has variance d_k, so scores
    #   grow like sqrt(d_k) and saturate softmax (zero gradient). note d_k,
    #   NOT d_model -- using d_model under-scales by sqrt(h).
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)   # (2,4,5,5)

    # axis=-1 normalizes over KEYS: each query spends one unit of attention
    # across all keys. axis=-2 would normalize per-key -- identical shape,
    # silent bug, model trains but underperforms.
    attention_weights = softmax(scores, axis=-1)                    # (2,4,5,5)

    # (B,h,L_q,L_k) @ (B,h,L_k,d_k) -> (B,h,L_q,d_k)
    # contracts the key axis: output row s = sum_t w[s,t] * v_t
    attention_output = np.matmul(attention_weights, V)              # (2,4,5,3)

    # ---- 4. merge heads --------------------------------------------------
    # transpose back: (B,h,L,d_k) -> (B,L,h,d_k)         (2,4,5,3)->(2,5,4,3)
    # reshape:        (B,L,h,d_k) -> (B,L,d_model)       (2,5,4,3)->(2,5,12)
    #   the swap MUST come first. reshaping straight from (B,h,L,d_k) would
    #   merge L into the feature axis and scramble positions.
    #   this reshape genuinely COPIES -- the array is non-contiguous after
    #   the transpose, and interleaving heads with positions is real data
    #   movement. it's the only allocation in the function besides the GEMMs.
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, d_model)                               # (2,5,12)

    # ---- 5. output projection -------------------------------------------
    # mixes information across heads -- without this, heads never interact.
    return np.dot(attention_output, W_o)                            # (2,5,12)