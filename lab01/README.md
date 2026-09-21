# Lab 01 — From Text Processing to Search

---

## 1. Giới thiệu tổng quan

Lab 01 tập trung nghiên cứu toàn diện quy trình chuyển đổi văn bản sang vector thưa (Sparse Vector Representations) và ứng dụng xây dựng hệ thống tìm kiếm tài liệu (Document Search) sử dụng TF-IDF trên tập ngữ liệu thực tế **30.000 documents**:
$$\text{Raw Text} \longrightarrow \text{Tokenization} \longrightarrow \text{CountVectorizer} \longrightarrow \text{TF-IDF} \longrightarrow \text{Cosine Similarity Search}$$

Mục tiêu chính của lab là kiểm chứng cách các quyết định trong khâu tiền xử lý (Preprocessing) ảnh hưởng trực tiếp đến không gian biểu diễn tài liệu và chất lượng truy xuất thông tin, từ đó chỉ ra các giới hạn cố hữu của phương pháp đối sánh từ vựng (Lexical Matching) và đặt nền móng cho biểu diễn ngữ nghĩa (Dense Semantic Embeddings).

---

## 2. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab01/` tuân thủ đúng cấu trúc chuẩn quy định tại **Mục 17 (Deliverables)** của đề bài:

```text
lab01/
├── README.md             # Tài liệu tổng quan và hướng dẫn chạy thực nghiệm
├── calculations.md       # Lời giải chi tiết các bài tính tay trên corpus đồ chơi (Part B: Ex 1 - 6)
├── prediction.md         # 3 dự đoán khoa học trước khi thực nghiệm trên corpus 30K (Part C)
├── implementation.py     # Cài đặt độc lập 6 hàm cốt lõi + Unit tests + So sánh scikit-learn (Part E)
├── experiments.ipynb     # Toàn bộ thực nghiệm trên 30K documents (Part D, F, G, H, I, J)
├── results.csv           # Bảng định lượng kết quả tìm kiếm (P@5, R@5, MRR) cho 5 queries mẫu
└── reflection.md         # Báo cáo phản tư, trả lời 7 câu Learning Check và khai báo AI (Part I, J, Sec 15, 16)
```

---

## 3. Hướng dẫn chạy và tái hiện kết quả

### 3.1. Thiết lập môi trường

Kích hoạt môi trường Python (Python 3.10+):
```bash
conda activate ml
pip install -r requirements.txt
```

### 3.2. Chạy kiểm thử Core Implementation (`implementation.py`)

Chạy script để thực thi các bài Unit Test độc lập và so sánh với thư viện `scikit-learn`:
```bash
python lab01/implementation.py
```

Kết quả mong đợi:
```text
[+] Running unit tests...
[+] test_build_vocabulary
[+] test_compute_counts
[+] test_compute_tf
[+] test_compute_idf
[+] test_compute_tfidf
[+] test_cosine_similarity
[+] All tests passed

[+] Compare with scikit-learn:
[*] Vocabulary: ['cat', 'dog', 'eats', 'fish', 'likes']
[*] IDF comparison & TF-IDF vector matching...
[+] Similarity(d1, d2) - My implementation: 0.244830
[+] Similarity(d1, d2) - Sklearn:           0.544328
```

### 3.3. Chạy thực nghiệm trên Notebook (`experiments.ipynb`)

Mở notebook bằng Jupyter Lab / Jupyter Notebook hoặc VS Code:
```bash
jupyter notebook lab01/experiments.ipynb
```
Notebook tích hợp đầy đủ các phần:
- **Part D (Section 7):** Thống kê kích thước và độ thưa ($S = 99.93\%$) của ma trận TF-IDF trên 30.000 documents.
- **Part F (Section 9):** So sánh 3 pipeline tiền xử lý (Minimal vs Normalized vs Subword/WordPiece).
- **Part G (Section 10):** Triển khai cỗ máy tìm kiếm dựa trên Cosine Similarity.
- **Part H (Section 11):** Đánh giá định lượng mô hình (xuất file `results.csv`).
- **Part I & J (Section 12 & 13):** Phân tích lỗi chi tiết cho 4 truy vấn và phân tích trường hợp thất bại quan trọng nhất (`heart attack treatment` vs `myocardial infarction therapy`).

---

## 4. Tóm tắt kết quả chính

### 4.1. Độ thưa của biểu diễn (Sparsity)
- Số lượng documents ($N$): **30.000**
- Kích thước từ vựng ($V$): **186.582** từ
- Tỷ lệ phần tử mang giá trị 0 ($S$): **99.9351%** (ma trận thưa cực đại).

### 4.2. Preprocessing Ablation
- **Pipeline A (Minimal):** 193.837 từ vựng, chứa nhiều stopwords làm loãng vector.
- **Pipeline B (Normalized):** 192.764 từ vựng, loại bỏ stopwords giúp giảm 35% số token, ma trận thưa hơn và tăng độ tương đồng với từ khóa chính.
- **Pipeline C (Extended - WordPiece Tokenizer):** Cố định 30.522 từ vựng, triệt tiêu hoàn toàn OOV ($0\%$).

### 4.3. Kết quả tìm kiếm (Retrieval Evaluation)

| Query | P@5 | R@5 | MRR | Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| `medical image classification` | 0.4000 | 1.0000 | 1.0000 | **Tốt:** Trùng từ khóa hiếm "classification" |
| `heart attack treatment` | 0.4000 | 1.0000 | 1.0000 | **Tốt:** Trùng cả cụm "heart" và "attack" |
| `deep learning healthcare` | 0.2000 | 0.5000 | 0.5000 | **Kém:** Bị nhiễu bởi văn bản chia thì "learned/learning" |
| `natural language processing` | 0.0000 | 0.0000 | 0.0000 | **Kém:** Bị bẻ gãy cụm từ thành các unigram rời rạc |
| `transformer language model` | 0.0000 | 0.0000 | 0.0000 | **Kém:** Nhầm sang máy biến áp (Polysemy) |
| **Trung bình (Mean)** | **0.2000** | **0.5000** | **0.5000** | — |

---

## 5. Kết luận & Hướng phát triển

Thực nghiệm chỉ ra hạn chế sống còn của **Lexical Matching**:
- Không thể nhận biết từ đồng nghĩa (ví dụ `heart attack` $\equiv$ `myocardial infarction` có $\cos = 0$).
- Không phân biệt được ngữ cảnh của từ đa nghĩa (`transformer`).
- Bị nhiễu khi gặp hiện tượng lặp từ bất thường (cần chuyển sang BM25 hoặc Dense Vector Embeddings như Sentence-BERT / Dense Retrieval).
