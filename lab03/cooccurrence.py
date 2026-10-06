import re
from collections import Counter
import numpy as np


def tokenize(text, lowercase=True, remove_punct=True):
    if isinstance(text, str):
        if lowercase:
            text = text.lower()
        if remove_punct:
            text = re.sub(r"[^\w\s]", " ", text)
        return text.split()
    elif isinstance(text, (list, tuple)):
        tokens = []
        for t in text:
            t_str = str(t).lower() if lowercase else str(t)
            if remove_punct:
                t_str = re.sub(r"[^\w\s]", " ", t_str)
            tokens.extend(t_str.split())
        return tokens
    return []


def build_vocabulary(corpus, min_freq=1, lowercase=True, remove_punct=True):
    word_counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc, lowercase=lowercase, remove_punct=remove_punct)
        word_counts.update(tokens)

    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    return sorted(vocab)


def build_cooccurrence_matrix(
    corpus, vocabulary, window_size=1, lowercase=True, remove_punct=True
):
    word2idx = {w: i for i, w in enumerate(vocabulary)}
    vocab_size = len(vocabulary)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    for doc in corpus:
        tokens = tokenize(doc, lowercase=lowercase, remove_punct=remove_punct)
        n = len(tokens)
        for i, target_word in enumerate(tokens):
            if target_word not in word2idx:
                continue
            target_idx = word2idx[target_word]

            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context_word = tokens[j]
                if context_word in word2idx:
                    context_idx = word2idx[context_word]
                    matrix[target_idx, context_idx] += 1.0

    return matrix


def cosine_similarity(vec1, vec2):
    v1 = np.asarray(vec1, dtype=np.float64)
    v2 = np.asarray(vec2, dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def most_similar(word, matrix, vocabulary, top_k=5):
    if word not in vocabulary:
        return []

    word2idx = {w: i for i, w in enumerate(vocabulary)}
    target_idx = word2idx[word]
    target_vec = matrix[target_idx]

    if np.linalg.norm(target_vec) == 0.0:
        return []

    similarities = []
    for other_word, other_idx in word2idx.items():
        if other_idx == target_idx:
            continue
        other_vec = matrix[other_idx]
        sim = cosine_similarity(target_vec, other_vec)
        similarities.append((other_word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def run_unit_tests():
    print("[+] Running unit tests for cooccurrence.py...")

    toy_corpus = [
        "the cat eats fish",
        "the cat likes milk",
        "the dog eats meat",
        "the dog likes fish",
    ]

    vocab = build_vocabulary(toy_corpus, min_freq=1)
    assert "cat" in vocab and "dog" in vocab and "fish" in vocab
    print("  [+] test_build_vocabulary passed.")

    target_vocab = ["cat", "dog", "eats", "likes", "fish", "milk", "meat"]
    matrix = build_cooccurrence_matrix(
        toy_corpus, target_vocab, window_size=1, remove_punct=False
    )
    assert matrix.shape == (7, 7)

    word2idx = {w: i for i, w in enumerate(target_vocab)}
    cat_vec = matrix[word2idx["cat"]]
    dog_vec = matrix[word2idx["dog"]]

    expected_cat_vec = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    expected_dog_vec = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(cat_vec, expected_cat_vec)
    assert np.allclose(dog_vec, expected_dog_vec)
    print("  [+] test_build_cooccurrence_matrix passed.")

    sim_cat_dog = cosine_similarity(cat_vec, dog_vec)
    assert np.isclose(sim_cat_dog, 1.0)
    sim_orthogonal = cosine_similarity([1, 0], [0, 1])
    assert np.isclose(sim_orthogonal, 0.0)
    print("  [+] test_cosine_similarity passed.")

    top_matches = most_similar("cat", matrix, target_vocab, top_k=2)
    assert len(top_matches) > 0
    assert top_matches[0][0] == "dog"
    assert np.isclose(top_matches[0][1], 1.0)
    print("  [+] test_most_similar passed.")

    print("[+] All unit tests passed successfully!\n")


if __name__ == "__main__":
    run_unit_tests()
