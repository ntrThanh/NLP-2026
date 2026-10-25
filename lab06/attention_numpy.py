import numpy as np


def naive_softmax(x, axis=-1):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_with_temperature(s, T=1.0):
    return stable_softmax(np.asarray(s, dtype=np.float64) / T)


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = K.shape[-1]

    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        # mask = True o nhung vi tri duoc phep attend
        scores = np.where(mask, scores, -1e9)

    attention_weights = stable_softmax(scores, axis=-1)
    context = attention_weights @ V
    return scores, attention_weights, context


def self_attention(X, W_q, W_k, W_v, mask=None):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    scores, attention_weights, context = scaled_dot_product_attention(Q, K, V, mask)
    return Q, K, V, scores, attention_weights, context


if __name__ == "__main__":
    x = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        print("Naive softmax :", naive_softmax(x))
    print("Stable softmax:", np.round(stable_softmax(x), 4))

    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 8))
    K = rng.normal(size=(7, 8))
    V = rng.normal(size=(7, 3))
    scores, A, C = scaled_dot_product_attention(Q, K, V)
    print("Scores shape:", scores.shape, "| Weights shape:", A.shape, "| Context shape:", C.shape)
    print("Row sums:", np.round(A.sum(axis=1), 6))
