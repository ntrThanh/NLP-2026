"""
Lab 01 — Part E: Core Implementation
Tự xây dựng phiên bản TF-IDF tối giản và Cosine Similarity không dùng thư viện trực tiếp.
Bao gồm:
  1. build_vocabulary()
  2. compute_counts()
  3. compute_tf()
  4. compute_idf()
  5. compute_tfidf()
  6. cosine_similarity()
  7. Bộ Unit tests kiểm chứng từng hàm (với câu lệnh assert)
  8. So sánh và giải thích chi tiết với Scikit-Learn TfidfVectorizer
"""

import math
from typing import Dict, List


# ==============================================================================
# 1. CORE FUNCTIONS (Theo mục 8.2 của đề bài)
# ==============================================================================

def build_vocabulary(documents: List[str]) -> List[str]:
    """
    Xây dựng từ điển (vocabulary) gồm các từ duy nhất từ danh sách documents,
    được sắp xếp theo thứ tự bảng chữ cái (alphabetical order).
    
    Args:
        documents: Danh sách các chuỗi văn bản.
        
    Returns:
        Danh sách các từ duy nhất đã được sắp xếp.
    """
    vocabulary = set()
    for doc in documents:
        for word in doc.split():
            vocabulary.add(word)
    return sorted(list(vocabulary))


def compute_counts(document: str, vocabulary: List[str]) -> Dict[str, int]:
    """
    Đếm số lần xuất hiện (raw count) của mỗi từ trong từ điển đối với một văn bản.
    
    Args:
        document: Chuỗi văn bản cần đếm.
        vocabulary: Danh sách từ điển.
        
    Returns:
        Dict dạng {word: count}.
    """
    counts = {w: 0 for w in vocabulary}
    for word in document.split():
        if word in counts:
            counts[word] += 1
    return counts


def compute_tf(count: Dict[str, int]) -> Dict[str, float]:
    """
    Tính Term Frequency (TF) theo công thức chuẩn lý thuyết:
        tf(t, d) = c(t, d) / sum(c(t', d))
        
    Args:
        count: Dict số lần xuất hiện của mỗi từ trong văn bản.
        
    Returns:
        Dict dạng {word: tf_value}. Nếu văn bản rỗng trả về 0.0 cho tất cả các từ.
    """
    total = sum(count.values())
    if total == 0:
        return {w: 0.0 for w in count}
    return {w: c / total for w, c in count.items()}


def compute_idf(documents: List[str], vocabulary: List[str]) -> Dict[str, float]:
    """
    Tính Inverse Document Frequency (IDF) theo công thức chuẩn:
        idf(t) = log(N / df(t))
        với N là tổng số văn bản, df(t) là số văn bản chứa từ t.
        
    Args:
        documents: Danh sách các văn bản trong corpus.
        vocabulary: Danh sách từ điển.
        
    Returns:
        Dict dạng {word: idf_value}.
    """
    n = len(documents)
    idf = {}
    for w in vocabulary:
        df = sum(1 for doc in documents if w in doc.split())
        idf[w] = math.log(n / df) if df > 0 else 0.0
    return idf


def compute_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Tính trọng số TF-IDF cho từng từ:
        tfidf(t, d) = tf(t, d) * idf(t)
        
    Args:
        tf: Dict TF của văn bản.
        idf: Dict IDF của corpus.
        
    Returns:
        Dict dạng {word: tfidf_value}.
    """
    return {w: tf[w] * idf.get(w, 0.0) for w in tf}


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Tính độ tương đồng Cosine giữa hai vector (dạng dict):
        cos(x, y) = (x . y) / (||x||_2 * ||y||_2)
        
    Args:
        vec1: Dict vector thứ nhất {word: weight}.
        vec2: Dict vector thứ hai {word: weight}.
        
    Returns:
        Giá trị cosine similarity nằm trong đoạn [0.0, 1.0].
    """
    dot_product = sum(vec1[w] * vec2[w] for w in vec1 if w in vec2)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ==============================================================================
# 2. UNIT TESTS (Theo mục 8.4 của đề bài)
# ==============================================================================

def test_build_vocabulary():
    """Kiểm tra hàm build_vocabulary."""
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = build_vocabulary(docs)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab, f"Sai vocab: {vocab} != {expected_vocab}"
    
    # Test văn bản rỗng
    assert build_vocabulary([]) == []
    print("  [PASS] test_build_vocabulary")


def test_compute_counts():
    """Kiểm tra hàm compute_counts."""
    vocab = ["cat", "dog", "eats", "fish", "likes"]
    doc = "cat eats fish"
    counts = compute_counts(doc, vocab)
    expected_counts = {"cat": 1, "dog": 0, "eats": 1, "fish": 1, "likes": 0}
    assert counts == expected_counts, f"Sai counts: {counts} != {expected_counts}"
    print("  [PASS] test_compute_counts")


def test_compute_tf():
    """Kiểm tra hàm compute_tf."""
    counts = {"cat": 1, "dog": 0, "eats": 1, "fish": 1, "likes": 0}
    tf = compute_tf(counts)
    expected_val = 1.0 / 3.0
    
    assert abs(tf["cat"] - expected_val) < 1e-9, f"Sai tf(cat): {tf['cat']}"
    assert abs(tf["dog"] - 0.0) < 1e-9, f"Sai tf(dog): {tf['dog']}"
    assert abs(sum(tf.values()) - 1.0) < 1e-9, f"Tổng TF khác 1.0: {sum(tf.values())}"
    
    # Test document rỗng
    empty_counts = {"cat": 0, "dog": 0}
    assert compute_tf(empty_counts) == {"cat": 0.0, "dog": 0.0}
    print("  [PASS] test_compute_tf")


def test_compute_idf():
    """Kiểm tra hàm compute_idf."""
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocab = ["cat", "dog", "eats", "fish", "likes"]
    idf = compute_idf(docs, vocab)
    
    # fish xuất hiện ở cả 3 docs -> df=3 -> log(3/3) = log(1) = 0.0
    assert abs(idf["fish"] - 0.0) < 1e-9, f"Sai idf(fish): {idf['fish']} != 0.0"
    
    # dog xuất hiện ở 1 doc -> df=1 -> log(3/1) = log(3)
    assert abs(idf["dog"] - math.log(3.0)) < 1e-9, f"Sai idf(dog): {idf['dog']}"
    
    # cat xuất hiện ở 2 docs -> df=2 -> log(3/2) = log(1.5)
    assert abs(idf["cat"] - math.log(1.5)) < 1e-9, f"Sai idf(cat): {idf['cat']}"
    print("  [PASS] test_compute_idf")


def test_compute_tfidf():
    """Kiểm tra hàm compute_tfidf."""
    tf = {"cat": 1/3, "dog": 0.0, "eats": 1/3, "fish": 1/3, "likes": 0.0}
    idf = {"cat": math.log(1.5), "dog": math.log(3.0), "eats": math.log(1.5), "fish": 0.0, "likes": math.log(3.0)}
    tfidf = compute_tfidf(tf, idf)
    
    expected_cat = (1/3) * math.log(1.5)
    assert abs(tfidf["cat"] - expected_cat) < 1e-9, f"Sai tfidf(cat): {tfidf['cat']}"
    assert abs(tfidf["fish"] - 0.0) < 1e-9, f"Sai tfidf(fish): {tfidf['fish']} != 0.0"
    assert abs(tfidf["dog"] - 0.0) < 1e-9, f"Sai tfidf(dog): {tfidf['dog']}"
    print("  [PASS] test_compute_tfidf")


def test_cosine_similarity():
    """Kiểm tra hàm cosine_similarity."""
    v1 = {"a": 1.0, "b": 0.0}
    v2 = {"a": 1.0, "b": 0.0}
    v3 = {"a": 0.0, "b": 1.0}
    v_zero = {"a": 0.0, "b": 0.0}
    
    # Vector giống nhau -> cos = 1.0
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-9, "Hai vector giống nhau phải có cosine = 1.0"
    
    # Vector trực giao -> cos = 0.0
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-9, "Hai vector trực giao phải có cosine = 0.0"
    
    # Vector zero -> cos = 0.0 (không crash chia cho 0)
    assert abs(cosine_similarity(v1, v_zero) - 0.0) < 1e-9, "Vector zero phải trả về 0.0"
    
    # Kiểm tra với D1 và D2 theo công thức lý thuyết
    idf_cat = math.log(1.5)
    idf_dog = math.log(3.0)
    idf_eats = math.log(1.5)
    
    tfidf1 = {"cat": (1/3)*idf_cat, "dog": 0.0, "eats": (1/3)*idf_eats, "fish": 0.0, "likes": 0.0}
    tfidf2 = {"cat": 0.0, "dog": (1/3)*idf_dog, "eats": (1/3)*idf_eats, "fish": 0.0, "likes": 0.0}
    
    expected_sim = idf_eats / (math.sqrt(2) * math.sqrt(idf_dog**2 + idf_eats**2))
    computed_sim = cosine_similarity(tfidf1, tfidf2)
    assert abs(computed_sim - expected_sim) < 1e-9, f"Sai similarity D1-D2: {computed_sim} != {expected_sim}"
    print("  [PASS] test_cosine_similarity")


def run_unit_tests():
    """Chạy toàn bộ các unit tests."""
    print("=" * 60)
    print("CHẠY CÁC UNIT TESTS THEO MỤC 8.4")
    print("=" * 60)
    test_build_vocabulary()
    test_compute_counts()
    test_compute_tf()
    test_compute_idf()
    test_compute_tfidf()
    test_cosine_similarity()
    print("=> TẤT CẢ UNIT TESTS ĐÃ VƯỢT QUA THÀNH CÔNG (ALL TESTS PASSED)!\n")


# ==============================================================================
# 3. SO SÁNH VỚI THƯ VIỆN SCIKIT-LEARN (Theo mục 8.5 của đề bài)
# ==============================================================================

def compare_with_reference():
    """
    So sánh cài đặt tự viết với TfidfVectorizer của scikit-learn trên corpus đồ chơi.
    Phân tích nguyên nhân khác biệt về mặt số học:
      1. Công thức IDF và smoothing
      2. Định nghĩa TF (tần suất tương đối vs raw count)
      3. Chuẩn hóa vector (L2 normalization)
    """
    docs = ["cat eats fish", "dog eats fish", "cat likes fish"]
    
    # 1. Triển khai tự viết (Student Implementation)
    vocab = build_vocabulary(docs)
    idf_student = compute_idf(docs, vocab)
    
    counts_d1 = compute_counts(docs[0], vocab)
    tf_d1 = compute_tf(counts_d1)
    tfidf_d1_student = compute_tfidf(tf_d1, idf_student)
    
    counts_d2 = compute_counts(docs[1], vocab)
    tf_d2 = compute_tf(counts_d2)
    tfidf_d2_student = compute_tfidf(tf_d2, idf_student)
    
    sim_d1_d2_student = cosine_similarity(tfidf_d1_student, tfidf_d2_student)
    
    # 2. Thư viện chuẩn Scikit-Learn (Reference Implementation)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer()
        X_sklearn = vec.fit_transform(docs).toarray()
        sklearn_vocab = vec.get_feature_names_out().tolist()
        sklearn_idf = dict(zip(sklearn_vocab, vec.idf_))
        sklearn_d1 = dict(zip(sklearn_vocab, X_sklearn[0]))
        sklearn_d2 = dict(zip(sklearn_vocab, X_sklearn[1]))
        
        # Cosine similarity bằng vector sklearn
        dot_sk = sum(sklearn_d1[w] * sklearn_d2[w] for w in sklearn_vocab)
        norm_sk1 = math.sqrt(sum(v**2 for v in sklearn_d1.values()))
        norm_sk2 = math.sqrt(sum(v**2 for v in sklearn_d2.values()))
        sim_sklearn = dot_sk / (norm_sk1 * norm_sk2) if norm_sk1 * norm_sk2 > 0 else 0.0
        
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    
    print("=" * 60)
    print("SO SÁNH: STUDENT IMPLEMENTATION vs SCIKIT-LEARN (MỤC 8.5)")
    print("=" * 60)
    print(f"Từ điển (Vocabulary): {vocab}")
    print("-" * 60)
    print(f"{'Từ':<8} | {'IDF (Tự viết)':<18} | {'IDF (Scikit-Learn)':<18}")
    print("-" * 60)
    for w in vocab:
        sk_val = f"{sklearn_idf.get(w, 0.0):.6f}" if has_sklearn else "N/A"
        print(f"{w:<8} | {idf_student[w]:<18.6f} | {sk_val:<18}")
        
    print("-" * 60)
    print("TF-IDF CỦA DOCUMENT 1 ('cat eats fish'):")
    print(f"{'Từ':<8} | {'TF-IDF (Tự viết)':<18} | {'TF-IDF (Scikit-Learn)':<18}")
    print("-" * 60)
    for w in vocab:
        sk_val = f"{sklearn_d1.get(w, 0.0):.6f}" if has_sklearn else "N/A"
        print(f"{w:<8} | {tfidf_d1_student[w]:<18.6f} | {sk_val:<18}")
        
    print("-" * 60)
    print(f"Cosine Similarity (D1, D2) - Tự viết:     {sim_d1_d2_student:.6f}")
    if has_sklearn:
        print(f"Cosine Similarity (D1, D2) - Scikit-Learn: {sim_sklearn:.6f}")
    print("=" * 60)
    
    print("\nGIẢI THÍCH SỰ KHÁC BIỆT THEO MỤC 8.5:")
    print("1. Khác nhau ở công thức IDF:")
    print("   - Tự viết dùng công thức nguyên bản: idf(t) = ln(N / df(t)).")
    print("     Do đó với từ 'fish' (xuất hiện ở 3/3 docs), idf(fish) = ln(3/3) = 0.0.")
    print("   - Scikit-Learn dùng công thức smooth_idf=True mặc định:")
    print("     idf(t) = ln((1 + N) / (1 + df(t))) + 1.")
    print("     Do đó với 'fish': idf = ln(4/4) + 1 = 1.0 (không bị bằng 0).")
    print()
    print("2. Khác nhau ở công thức TF:")
    print("   - Tự viết dùng tần suất tương đối: tf(t, d) = c(t, d) / tổng_từ_trong_doc (ở đây là 1/3).")
    print("   - Scikit-Learn mặc định dùng raw count: tf(t, d) = c(t, d) (ở đây là 1).")
    print()
    print("3. Khác nhau ở chuẩn hóa vector (L2 Normalization):")
    print("   - Scikit-Learn mặc định áp dụng chuẩn hóa L2 cho từng hàng vector: v = v / ||v||_2.")
    print("   - Tự viết giữ vector raw (tf * idf) và chỉ thực hiện chuẩn hóa bên trong hàm cosine_similarity.")
    print("=" * 60)


def main():
    run_unit_tests()
    compare_with_reference()


if __name__ == "__main__":
    main()
