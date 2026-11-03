"""
Module: positional_encoding.py
Mục đích:
- Cài đặt Sinusoidal Positional Encoding theo đúng chuẩn kiến trúc Transformer (Vaswani et al., 2017).
- Thực hiện Experiment 1 (Mục 11): So sánh chuỗi ["I", "love", "NLP"] và ["NLP", "love", "I"].
- Thực hiện Permutation Experiment (Mục 12): So sánh S1 = "dog bites man" và S2 = "man bites dog"
  với và không với Positional Encoding để chứng minh vai trò thiết yếu của thông tin vị trí.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def stable_softmax(x, axis=-1):
    """Tính Softmax ổn định về mặt số học."""
    x = np.asarray(x, dtype=np.float64)
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PositionalEncoding:
    """Sinusoidal Positional Encoding cho Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len

        # Khởi tạo ma trận PE kích thước (max_len, d_model)
        pe = np.zeros((max_len, d_model), dtype=np.float64)
        position = np.arange(0, max_len, dtype=np.float64)[:, np.newaxis]

        # div_term = 10000^(2i / d_model) = exp(2i * -log(10000.0) / d_model)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def get_encoding(self, seq_len: int):
        """Lấy ma trận PE cho độ dài sequence seq_len."""
        return self.pe[:seq_len, :]


def self_attention(X, W_q=None, W_k=None, W_v=None):
    """
    Tính Self-Attention chuẩn.
    Nếu W_q, W_k, W_v không được cung cấp, sử dụng phép chiếu đồng nhất (Identity).
    """
    T, d = X.shape
    Q = X @ W_q if W_q is not None else X
    K = X @ W_k if W_k is not None else X
    V = X @ W_v if W_v is not None else X

    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    attention_weights = stable_softmax(scores, axis=-1)
    context = attention_weights @ V
    return scores, attention_weights, context


def cosine_similarity(v1, v2):
    """Tính cosine similarity giữa 2 vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / (norm + 1e-9)


def run_all_experiments():
    np.random.seed(42)
    d_model = 4
    pe_module = PositionalEncoding(d_model=d_model, max_len=20)

    print("=" * 60)
    print("MỤC 10: TÍNH TOÁN POSITIONAL ENCODING (d=4, pos=0, 1, 2)")
    print("=" * 60)
    pe_small = pe_module.get_encoding(3)
    for pos in range(3):
        print(f"pos = {pos}: {pe_small[pos]}")

    print("\n" + "=" * 60)
    print("MỤC 11: EXPERIMENT 1 — ['I', 'love', 'NLP'] vs ['NLP', 'love', 'I']")
    print("=" * 60)
    # Giả lập embedding ngẫu nhiên cho từ điển
    vocab = {"I": 0, "love": 1, "NLP": 2}
    embedding_table = {
        "I": np.array([1.0, 0.0, 1.0, 0.0]),
        "love": np.array([0.0, 2.0, 0.0, 1.0]),
        "NLP": np.array([1.0, 1.0, 0.0, 1.0]),
    }

    seq1 = ["I", "love", "NLP"]
    seq2 = ["NLP", "love", "I"]

    X1 = np.stack([embedding_table[w] for w in seq1])  # (3, 4)
    X2 = np.stack([embedding_table[w] for w in seq2])  # (3, 4)

    # 1. Không có Positional Encoding
    _, A1_no_pe, Z1_no_pe = self_attention(X1)
    _, A2_no_pe, Z2_no_pe = self_attention(X2)

    # Mean pooling cho cả câu
    sent_rep1_no_pe = np.mean(Z1_no_pe, axis=0)
    sent_rep2_no_pe = np.mean(Z2_no_pe, axis=0)

    # 2. Có Positional Encoding
    PE3 = pe_module.get_encoding(3)
    X1_pe = X1 + PE3
    X2_pe = X2 + PE3

    _, A1_pe, Z1_pe = self_attention(X1_pe)
    _, A2_pe, Z2_pe = self_attention(X2_pe)

    sent_rep1_pe = np.mean(Z1_pe, axis=0)
    sent_rep2_pe = np.mean(Z2_pe, axis=0)

    print(f"Không có PE - Cosine similarity giữa seq1 và seq2 (sau mean pooling): {cosine_similarity(sent_rep1_no_pe, sent_rep2_no_pe):.4f}")
    print(f"Không có PE - Khoảng cách Euclidean: {np.linalg.norm(sent_rep1_no_pe - sent_rep2_no_pe):.4f}")
    print(f"Có PE       - Cosine similarity giữa seq1 và seq2 (sau mean pooling): {cosine_similarity(sent_rep1_pe, sent_rep2_pe):.4f}")
    print(f"Có PE       - Khoảng cách Euclidean: {np.linalg.norm(sent_rep1_pe - sent_rep2_pe):.4f}")

    print("\n" + "=" * 60)
    print("MỤC 12: PERMUTATION EXPERIMENT (S1='dog bites man' vs S2='man bites dog')")
    print("=" * 60)
    vocab_perm = {
        "dog": np.array([1.2, 0.3, -0.5, 0.8]),
        "bites": np.array([0.1, 1.5, 0.2, -0.4]),
        "man": np.array([-0.8, 0.2, 1.4, 0.5]),
    }
    S1_words = ["dog", "bites", "man"]
    S2_words = ["man", "bites", "dog"]

    X_S1 = np.stack([vocab_perm[w] for w in S1_words])
    X_S2 = np.stack([vocab_perm[w] for w in S2_words])

    # Experiment A: Không có PE
    _, A_S1_no_pe, Z_S1_no_pe = self_attention(X_S1)
    _, A_S2_no_pe, Z_S2_no_pe = self_attention(X_S2)

    rep_S1_no_pe = np.mean(Z_S1_no_pe, axis=0)
    rep_S2_no_pe = np.mean(Z_S2_no_pe, axis=0)

    # Experiment B: Có PE
    X_S1_pe = X_S1 + PE3
    X_S2_pe = X_S2 + PE3

    _, A_S1_pe, Z_S1_pe = self_attention(X_S1_pe)
    _, A_S2_pe, Z_S2_pe = self_attention(X_S2_pe)

    rep_S1_pe = np.mean(Z_S1_pe, axis=0)
    rep_S2_pe = np.mean(Z_S2_pe, axis=0)

    print("Experiment A (Không có Positional Encoding):")
    print(f"  Representation của toàn câu S1: {rep_S1_no_pe}")
    print(f"  Representation của toàn câu S2: {rep_S2_no_pe}")
    print(f"  Cosine Similarity(S1, S2): {cosine_similarity(rep_S1_no_pe, rep_S2_no_pe):.6f}")
    print(f"  Euclidean Distance(S1, S2): {np.linalg.norm(rep_S1_no_pe - rep_S2_no_pe):.6e}")
    print("  -> Nhận xét: Hai câu mang ý nghĩa trái ngược nhau lại có representation GIỐNG NHAU 100%!")

    print("\nExperiment B (Có Positional Encoding):")
    print(f"  Representation của toàn câu S1: {rep_S1_pe}")
    print(f"  Representation của toàn câu S2: {rep_S2_pe}")
    print(f"  Cosine Similarity(S1, S2): {cosine_similarity(rep_S1_pe, rep_S2_pe):.6f}")
    print(f"  Euclidean Distance(S1, S2): {np.linalg.norm(rep_S1_pe - rep_S2_pe):.6f}")
    print("  -> Nhận xét: Nhờ Positional Encoding, representation của hai câu đã TÁCH BIỆT RÕ RÀNG!")

    # Vẽ và lưu biểu đồ trực quan hóa
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    def plot_matrix(ax, mat, xticks, yticks, title, cmap="Blues"):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_xticks(range(len(xticks)))
        ax.set_yticks(range(len(yticks)))
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > np.mean(mat) else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 1. Positional Encoding Matrix Heatmap
    pe_large = PositionalEncoding(d_model=16, max_len=30).get_encoding(30)
    im0 = axes[0, 0].imshow(pe_large, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("1. Sinusoidal PE Heatmap (30 pos x 16 dim)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Dimension i")
    axes[0, 0].set_ylabel("Position pos")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Attention Matrix S1 Không PE
    plot_matrix(axes[0, 1], A_S1_no_pe, S1_words, S1_words, "2. S1 ('dog bites man') Attention [No PE]", "Blues")

    # 3. Attention Matrix S2 Không PE
    plot_matrix(axes[0, 2], A_S2_no_pe, S2_words, S2_words, "3. S2 ('man bites dog') Attention [No PE]", "Blues")

    # 4. Attention Matrix S1 Có PE
    plot_matrix(axes[1, 0], A_S1_pe, S1_words, S1_words, "4. S1 ('dog bites man') Attention [With PE]", "Oranges")

    # 5. Attention Matrix S2 Có PE
    plot_matrix(axes[1, 1], A_S2_pe, S2_words, S2_words, "5. S2 ('man bites dog') Attention [With PE]", "Oranges")

    # 6. Biểu đồ so sánh khoảng cách và độ tương đồng
    metrics = ["Euclidean Distance", "1 - Cosine Sim"]
    no_pe_vals = [np.linalg.norm(rep_S1_no_pe - rep_S2_no_pe), 1.0 - cosine_similarity(rep_S1_no_pe, rep_S2_no_pe)]
    pe_vals = [np.linalg.norm(rep_S1_pe - rep_S2_pe), 1.0 - cosine_similarity(rep_S1_pe, rep_S2_pe)]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 2].bar(x - width / 2, no_pe_vals, width, label="No PE", color="steelblue")
    axes[1, 2].bar(x + width / 2, pe_vals, width, label="With PE", color="coral")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics, fontweight="bold")
    axes[1, 2].set_ylabel("Difference Magnitude")
    axes[1, 2].set_title("6. Distinction between S1 and S2", fontsize=11, fontweight="bold")
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", linestyle="--", alpha=0.5)

    save_path = os.path.join(os.path.dirname(__file__), "positional_encoding.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[THÀNH CÔNG] Đã lưu biểu đồ thực nghiệm tại: {save_path}")


if __name__ == "__main__":
    run_all_experiments()
