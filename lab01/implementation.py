import math
from typing import Dict, List


def build_vocabulary(documents: List[str]) -> List[str]:
    vocabulary = set()
    for doc in documents:
        for word in doc.split():
            vocabulary.add(word)
    return sorted(list(vocabulary))


def compute_counts(document: str, vocabulary: List[str]) -> Dict[str, int]:
    counts = {w: 0 for w in vocabulary}
    for word in document.split():
        if word in counts:
            counts[word] += 1
    return counts


def compute_tf(count: Dict[str, int]) -> Dict[str, float]:
    total = sum(count.values())
    if total == 0:
        return {w: 0.0 for w in count}
    return {w: c / total for w, c in count.items()}


def compute_idf(documents: List[str], vocabulary: List[str]) -> Dict[str, float]:
    n = len(documents)
    idf = {}
    for w in vocabulary:
        df = sum(1 for doc in documents if w in doc.split())
        idf[w] = math.log(n / df) if df > 0 else 0.0
    return idf


def compute_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    return {w: tf[w] * idf.get(w, 0.0) for w in tf}


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    dot_product = sum(vec1[w] * vec2[w] for w in vec1 if w in vec2)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ==========================================
# Unit tests
# ==========================================

def test_build_vocabulary():
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = build_vocabulary(docs)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab
    assert build_vocabulary([]) == []
    print("  [PASS] test_build_vocabulary")


def test_compute_counts():
    vocab = ["cat", "dog", "eats", "fish", "likes"]
    doc = "cat eats fish"
    counts = compute_counts(doc, vocab)
    expected_counts = {"cat": 1, "dog": 0, "eats": 1, "fish": 1, "likes": 0}
    assert counts == expected_counts
    print("  [PASS] test_compute_counts")


def test_compute_tf():
    counts = {"cat": 1, "dog": 0, "eats": 1, "fish": 1, "likes": 0}
    tf = compute_tf(counts)
    expected_val = 1.0 / 3.0
    
    assert abs(tf["cat"] - expected_val) < 1e-9
    assert abs(tf["dog"] - 0.0) < 1e-9
    assert abs(sum(tf.values()) - 1.0) < 1e-9
    
    empty_counts = {"cat": 0, "dog": 0}
    assert compute_tf(empty_counts) == {"cat": 0.0, "dog": 0.0}
    print("  [PASS] test_compute_tf")


def test_compute_idf():
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = ["cat", "dog", "eats", "fish", "likes"]
    idf = compute_idf(docs, vocab)
    
    assert abs(idf["fish"] - 0.0) < 1e-9
    assert abs(idf["dog"] - math.log(3.0)) < 1e-9
    assert abs(idf["cat"] - math.log(1.5)) < 1e-9
    print("  [PASS] test_compute_idf")


def test_compute_tfidf():
    tf = {"cat": 1/3, "dog": 0.0, "eats": 1/3, "fish": 1/3, "likes": 0.0}
    idf = {"cat": math.log(1.5), "dog": math.log(3.0), "eats": math.log(1.5), "fish": 0.0, "likes": math.log(3.0)}
    tfidf = compute_tfidf(tf, idf)
    
    expected_cat = (1/3) * math.log(1.5)
    assert abs(tfidf["cat"] - expected_cat) < 1e-9
    assert abs(tfidf["fish"] - 0.0) < 1e-9
    assert abs(tfidf["dog"] - 0.0) < 1e-9
    print("  [PASS] test_compute_tfidf")


def test_cosine_similarity():
    v1 = {"a": 1.0, "b": 0.0}
    v2 = {"a": 1.0, "b": 0.0}
    v3 = {"a": 0.0, "b": 1.0}
    v_zero = {"a": 0.0, "b": 0.0}
    
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-9
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-9
    assert abs(cosine_similarity(v1, v_zero) - 0.0) < 1e-9
    
    idf_cat = math.log(1.5)
    idf_dog = math.log(3.0)
    idf_eats = math.log(1.5)
    
    tfidf1 = {"cat": (1/3)*idf_cat, "dog": 0.0, "eats": (1/3)*idf_eats, "fish": 0.0, "likes": 0.0}
    tfidf2 = {"cat": 0.0, "dog": (1/3)*idf_dog, "eats": (1/3)*idf_eats, "fish": 0.0, "likes": 0.0}
    
    expected_sim = idf_eats / (math.sqrt(2) * math.sqrt(idf_dog**2 + idf_eats**2))
    computed_sim = cosine_similarity(tfidf1, tfidf2)
    assert abs(computed_sim - expected_sim) < 1e-9
    print("  [PASS] test_cosine_similarity")


def run_unit_tests():
    print("=" * 50)
    print("RUNNING UNIT TESTS")
    print("=" * 50)
    test_build_vocabulary()
    test_compute_counts()
    test_compute_tf()
    test_compute_idf()
    test_compute_tfidf()
    test_cosine_similarity()
    print("=> ALL TESTS PASSED!\n")


def compare_with_reference():
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    
    vocab = build_vocabulary(docs)
    idf_student = compute_idf(docs, vocab)
    
    tf_d1 = compute_tf(compute_counts(docs[0], vocab))
    tfidf_d1_student = compute_tfidf(tf_d1, idf_student)
    
    tf_d2 = compute_tf(compute_counts(docs[1], vocab))
    tfidf_d2_student = compute_tfidf(tf_d2, idf_student)
    
    sim_d1_d2 = cosine_similarity(tfidf_d1_student, tfidf_d2_student)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer()
        X_sklearn = vec.fit_transform(docs).toarray()
        sklearn_vocab = vec.get_feature_names_out().tolist()
        sklearn_idf = dict(zip(sklearn_vocab, vec.idf_))
        sklearn_d1 = dict(zip(sklearn_vocab, X_sklearn[0]))
        sklearn_d2 = dict(zip(sklearn_vocab, X_sklearn[1]))
        
        dot_sk = sum(sklearn_d1[w] * sklearn_d2[w] for w in sklearn_vocab)
        norm_sk1 = math.sqrt(sum(v**2 for v in sklearn_d1.values()))
        norm_sk2 = math.sqrt(sum(v**2 for v in sklearn_d2.values()))
        sim_sklearn = dot_sk / (norm_sk1 * norm_sk2) if norm_sk1 * norm_sk2 > 0 else 0.0
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    
    print("=" * 50)
    print("COMPARE WITH SCIKIT-LEARN")
    print("=" * 50)
    print(f"Vocabulary: {vocab}\n")
    print(f"{'Word':<8} | {'My IDF':<12} | {'Sklearn IDF':<12}")
    print("-" * 38)
    for w in vocab:
        sk_val = f"{sklearn_idf.get(w, 0.0):.6f}" if has_sklearn else "N/A"
        print(f"{w:<8} | {idf_student[w]:<12.6f} | {sk_val:<12}")
        
    print("\nTF-IDF d1 ('cat eats fish'):")
    print(f"{'Word':<8} | {'My TF-IDF':<12} | {'Sklearn TF-IDF':<14}")
    print("-" * 40)
    for w in vocab:
        sk_val = f"{sklearn_d1.get(w, 0.0):.6f}" if has_sklearn else "N/A"
        print(f"{w:<8} | {tfidf_d1_student[w]:<12.6f} | {sk_val:<14}")
        
    print(f"\nSimilarity(d1, d2) - My implementation: {sim_d1_d2:.6f}")
    if has_sklearn:
        print(f"Similarity(d1, d2) - Sklearn:           {sim_sklearn:.6f}")
    print("=" * 50)


def main():
    run_unit_tests()
    compare_with_reference()


if __name__ == "__main__":
    main()
