# Lab 02 — Language Models: N-gram, Smoothing and Perplexity

---

## 1. Giới thiệu tổng quan

Tiếp nối **Lab 01** (biểu diễn văn bản dưới dạng vector đếm thưa TF-IDF phục vụ bài toán tìm kiếm), **Lab 02** chuyển dịch sang khía cạnh xác suất của ngôn ngữ tự nhiên: **Mô hình hóa ngôn ngữ (Language Modeling)**.

Mục tiêu cốt lõi của Language Model là gán một phân phối xác suất hợp lý cho chuỗi từ:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

Áp dụng giả định Markov bậc $n-1$, mô hình N-gram Language Model xấp xỉ xác suất từ hiện tại chỉ dựa trên $n-1$ từ đứng trước:
- **Unigram ($n=1$):** Độc lập hoàn toàn, $P(w_t)$.
- **Bigram ($n=2$):** Phụ thuộc 1 từ trước đó, $P(w_t \mid w_{t-1})$.
- **Trigram ($n=3$):** Phụ thuộc 2 từ trước đó, $P(w_t \mid w_{t-2}, w_{t-1})$.

Lab này tập trung nghiên cứu toàn diện từ lý thuyết tính toán, cài đặt độc lập từ đầu (from scratch), giải quyết vấn đề xác suất bằng 0 (**Zero Probability Problem**) bằng phương pháp làm mịn **Laplace Smoothing**, đánh giá chất lượng bằng thước đo **Perplexity**, và ứng dụng vào hai bài toán thực tế: **Next-Word Prediction** và **Sentence Ranking**.

---

## 2. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab02/` tuân thủ nghiêm ngặt cấu trúc quy định tại **Mục 27 (Deliverables)** của đề bài:

```text
lab02/
├── README.md                          # Tài liệu tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md                    # Lời giải chi tiết các bài tính tay (Mục 7, 9, 11, 18)
├── prediction.md                      # 5 dự đoán khoa học trước thực nghiệm & đối chiếu kết quả (Mục 12)
├── ngram_lm.py                        # Cài đặt độc lập toàn bộ các hàm và class NGramLanguageModel (Mục 14)
├── experiments.ipynb                  # Toàn bộ thực nghiệm trên 30.000 documents (Mục 13, 15, 16, 19, 20, 21, 22, 23)
├── results.csv                        # Bảng định lượng Perplexity trên tập Train/Validation/Test (Mục 19)
├── error_analysis.md                  # Phân tích lỗi chi tiết cho 2 ca đoán đúng và 2 ca đoán sai (Mục 22)
├── reflection.md                      # Trả lời 7 câu hỏi phản tư, AI policy statement & Learning check (Mục 24, 25, 26)
├── NguyenTrongThanh_calculations.pdf  # Bản scan bài làm viết tay 9 trang chính chủ (Lý thuyết, Tính toán, Dự đoán)
└── link_submit.txt                    # Đường dẫn tới thư mục nộp bài trên GitHub
```

---

## 3. Hướng dẫn chạy và tái hiện kết quả

### 3.1. Thiết lập môi trường

Yêu cầu môi trường Python 3.10+ với các thư viện cơ bản (`numpy`, `pandas`, `matplotlib`):
```bash
conda activate ml
pip install numpy pandas matplotlib
```

### 3.2. Chạy kiểm thử mã nguồn độc lập (`ngram_lm.py`)

Chạy script để kiểm tra các hàm tính toán xác suất, đếm n-gram, tính log-probability và kiểm thử class `NGramLanguageModel` trên toy corpus:
```bash
python3 lab02/ngram_lm.py
```

Kết quả mong đợi:
```text
Vocabulary: ['cat', 'dog', 'eats', 'fish', 'likes', 'meat', 'the']
P(the): 0.25
P(cat|the): 0.6666666666666666
P(dog|the): 0.3333333333333333
P(eats|the cat): 0.5
Sentence prob (bi MLE): 0.041666666666666664
Sentence log prob (bi MLE): -3.1780538303479458
P_laplace(cat|the): 0.3
P_laplace(eats|the): 0.1
Next words after 'the': [('cat', 0.6666666666666666), ('dog', 0.3333333333333333)]
```

### 3.3. Chạy thực nghiệm trên Notebook (`experiments.ipynb`)

Mở notebook bằng Jupyter Lab / Jupyter Notebook hoặc VS Code:
```bash
jupyter notebook lab02/experiments.ipynb
```

Notebook tích hợp đầy đủ các phần:
- **Mục 13:** Thống kê tập ngữ liệu thực tế 30.000 documents (Vocabulary, Unique n-grams, tỷ lệ Singletons).
- **Mục 14 & 15:** Kiểm chứng cài đặt N-gram LM và tính toán an toàn với Log-probability (chống underflow).
- **Mục 16:** So sánh trực tiếp mô hình MLE vs Laplace Smoothing và hiện tượng Zero Probability trên tập Valid/Test.
- **Mục 19:** Đánh giá Perplexity toàn diện trên tập Train, Validation, Test và xuất file `results.csv`.
- **Mục 20:** Ứng dụng dự đoán từ tiếp theo (Next-Word Prediction) trên 5 ngữ cảnh.
- **Mục 21:** Ứng dụng xếp hạng câu (Sentence Ranking) cho các đoạn văn bản tiếp nối.
- **Mục 22:** Phân tích định tính lỗi dự đoán (Error Analysis).
- **Mục 23:** Khảo sát ảnh hưởng của chiều dài ngữ cảnh (Context Length: Unigram $\to$ Bigram $\to$ Trigram).
- **Mục 24, 25, 26:** Phản tư, tuyên bố sử dụng AI và chuẩn bị câu hỏi vấn đáp cá nhân.

---

## 4. Tóm tắt kết quả thực nghiệm chính

### 4.1. Thống kê tập ngữ liệu (Corpus Statistics & Sparsity)

- Số lượng documents ($N$): **30.000**
- Kích thước từ vựng ($|V|$): **186.618** từ
- Tổng số câu: **295.424** câu

| Mô hình | Số lượng Unique N-grams | Số lượng Singletons (xuất hiện 1 lần) | Tỷ lệ Singletons (%) |
| :--- | :---: | :---: | :---: |
| **Unigram** | 186.618 | 88.482 | **47.41%** |
| **Bigram** | 2.739.018 | 1.953.031 | **71.30%** |
| **Trigram** | 6.404.589 | 5.521.625 | **86.21%** |

> **Nhận xét:** Khi bậc $n$ tăng, không gian tổ hợp bùng nổ dẫn đến hiện tượng thưa dữ liệu cực kỳ trầm trọng (86.21% trigram chỉ xuất hiện đúng 1 lần trong cả 30.000 tài liệu).

### 4.2. So sánh MLE vs Laplace Smoothing

| Mô hình | Train PPL | Validation PPL | Test PPL |
| :--- | :---: | :---: | :---: |
| **Bigram MLE** | 209.81 | **$\infty$** | **$\infty$** |
| **Bigram Laplace** | 5.351,49 | 8.152,02 | 6.703,09 |
| **Trigram MLE** | 19.71 | **$\infty$** | **$\infty$** |
| **Trigram Laplace** | 27.462,79 | 49.395,52 | 43.047,01 |

> **Nhận xét:**
> - Mô hình MLE không thể đánh giá được trên tập dữ liệu chưa từng thấy (Valid/Test PPL = $\infty$) do gặp phải sự kiện có xác suất bằng 0.
> - Laplace Smoothing giải quyết triệt để lỗi vô cùng, cho phép mô hình hoạt động ổn định trên câu mới.

### 4.3. Bảng kết quả Perplexity (`results.csv`)

| Mô hình | Train PPL | Validation PPL | Test PPL |
| :--- | :---: | :---: | :---: |
| **Unigram** | 1.915,23 | 2.279,18 | **2.050,78** |
| **Bigram** | 5.351,49 | 8.152,02 | 6.703,09 |
| **Trigram** | 27.462,79 | 49.395,52 | 43.047,01 |

> **Hiện tượng quan trọng:** 
> Do từ vựng quá lớn ($|V| = 186.618$), mẫu số làm mịn Laplace ($C(h) + |V|$) đã phạt cực nặng các mô hình bậc cao khi gặp context mới. Điều này dẫn tới nghịch lý: **Unigram lại có Perplexity trên Test set tốt hơn Bigram và Trigram khi dùng Add-1 Smoothing**. Đây là minh chứng rõ ràng cho thấy Add-1 smoothing không phù hợp cho từ vựng lớn và cần các kỹ thuật làm mịn tiên tiến hơn như Kneser-Ney hoặc chuyển sang Neural LM.

### 4.4. Ứng dụng Next-Word Prediction & Sentence Ranking

1. **Next-Word Prediction (5 Contexts):**
   - `natural language` $\to$ `processing` (Chính xác, thuật ngữ cố định)
   - `one of` $\to$ `the` (Chính xác, cấu trúc ngữ pháp phổ biến)
   - `the cat` $\to$ `s` (Lỗi do domain mismatch và tokenizer tách sở hữu cách)
   - `machine learning` $\to$ `and` (Lỗi do stopword chiếm ưu thế)
   - `artificial intelligence` $\to$ `ai`

2. **Sentence Ranking (Context: `machine learning`):**
   - Xếp hạng 1: Candidate C: `studies language models` (Log-prob: $-35.35$, Prob: $4.43 \times 10^{-16}$)
   - Xếp hạng 2: Candidate B: `banana computer quickly` (Log-prob: $-36.04$, Prob: $2.22 \times 10^{-16}$)
   - Xếp hạng 3: Candidate A: `is useful for NLP` (Log-prob: $-36.88$, do dài 4 từ nên bị phạt tích xác suất, nhưng xét về Perplexity/độ bối rối trên từng từ thì Candidate A lại đạt PPL thấp nhất $= 10.096$).

---

## 5. Kết luận & Cầu nối sang Neural NLP

Thực nghiệm Lab 02 làm sáng tỏ các bản chất cốt lõi:
1. **N-gram LM hoạt động dựa trên thống kê đếm thuần túy:** Phụ thuộc hoàn toàn vào sự trùng khớp bề mặt từ ngữ (Lexical Matching).
2. **Hạn chế cố hữu:** 
   - **Data Sparsity:** Không thể khái quát hóa các từ đồng nghĩa hoặc ngữ cảnh tương đương.
   - **Giới hạn Markov:** Hoàn toàn mất trí nhớ sau 2 từ (đối với Trigram).
3. **Chuyển tiếp tự nhiên sang Lab 03 & Neural LM:**
   - Từ biểu diễn đếm rời rạc $\to$ Biểu diễn vector liên tục (**Word Embeddings: Word2Vec, GloVe**).
   - Từ cửa sổ Markov cố định $\to$ Mạng hồi quy (**RNN/LSTM**) và cơ chế chú ý (**Transformer/Self-Attention**) ghi nhớ ngữ cảnh không giới hạn.
