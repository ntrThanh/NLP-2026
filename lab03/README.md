# Lab 03 — Word Representations and Embeddings: From Sparse Representations to Dense Word Embeddings

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng  
Giảng viên / TA: Phạm Ngọc Hải  
Học kỳ: Học kỳ I - 2026  
Hạn nộp: 23:59 - 07/10/2026  

---

## 1. Giới thiệu tổng quan

Tiếp nối **Lab 01** (biểu diễn văn bản dạng thưa TF-IDF) và **Lab 02** (mô hình ngôn ngữ thống kê N-gram dựa trên tần số xuất hiện), **Lab 03** tập trung vào bước chuyển mình quan trọng của NLP: **Biểu diễn từ dạng vector đặc (Dense Word Embeddings)**.

Các phương pháp đếm truyền thống gặp phải hạn chế lớn: hai từ đồng nghĩa (ví dụ `doctor` và `physician`) có các vector one-hot hoặc TF-IDF hoàn toàn trực giao với nhau ($cosine = 0$). Dựa trên **Giả thuyết phân phối (Distributional Hypothesis)**: *"Words that occur in similar contexts tend to have similar meanings"*, Lab 03 nghiên cứu:
1. Xây dựng ma trận từ - ngữ cảnh (Word-Context Matrix) từ số liệu đồng xuất hiện cục bộ.
2. Cài đặt độc lập từ đầu (from scratch) các hàm cốt lõi: `tokenize`, `build_vocabulary`, `build_cooccurrence_matrix`, `cosine_similarity`, `most_similar`.
3. Huấn luyện mô hình mạng nơ-ron **Word2Vec** (kiến trúc CBOW) trên tập ngữ liệu thực tế.
4. Khảo sát ảnh hưởng của các siêu tham số quan trọng: kích thước cửa sổ ngữ cảnh (`window` $\in \{2, 5, 10\}$) và số chiều vector (`dimension` $\in \{50, 100, 300\}$).
5. Đánh giá định lượng qua các bài toán **Word Similarity** và **Word Analogy** ($king - man + woman \approx queen$).
6. Xây dựng ứng dụng tìm kiếm ngữ nghĩa (**Semantic Search**) và phân tích giới hạn của biểu diễn từ tĩnh trước hiện tượng từ đa nghĩa (**Polysemy**).

---

## 2. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab03/` tuân thủ nghiêm ngặt quy định tại Mục 30 của đề bài W3.pdf:

```text
lab03/
├── README.md             # Báo cáo tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md       # Lời giải chi tiết các bài tính toán lý thuyết
├── prediction.md         # 4 dự đoán khoa học trước khi thực nghiệm (Mục 9)
├── cooccurrence.py       # Cài đặt độc lập mã nguồn cốt lõi từ đầu (Mục 11)
├── experiments.ipynb     # Toàn bộ thực nghiệm có output đầy đủ (Mục 10 đến 24)
├── word_embedding.ipynb  # Bản sao đồng bộ của notebook thực nghiệm
├── results.csv           # Bảng số liệu định lượng về Window, Dimension và Similarity
├── error_analysis.md     # Phân tích 3 trường hợp đoán đúng và 3 trường hợp sai/bất ngờ (Mục 25)
├── reflection.md         # Phản tư về từ đa nghĩa bank, so sánh Transformer và ôn tập vấn đáp
└── W3.pdf                # Đề bài gốc của môn học
```

---

## 3. Hướng dẫn chạy và tái hiện kết quả

### 3.1. Thiết lập môi trường
Yêu cầu môi trường Python 3.10+ với các thư viện cơ bản (`numpy`, `pandas`, `matplotlib`, `gensim`):
```bash
pip install numpy pandas matplotlib gensim
```

### 3.2. Chạy kiểm thử mã nguồn độc lập (`cooccurrence.py`)
Script chạy các hàm xây dựng từ điển, lập ma trận đồng xuất hiện và tính tương đồng trên toy corpus:
```bash
python3 lab03/cooccurrence.py
```
Kết quả kiểm thử thành công:
```text
[+] Running unit tests for cooccurrence.py...
  [+] test_build_vocabulary passed.
  [+] test_build_cooccurrence_matrix passed.
  [+] test_cosine_similarity passed.
  [+] test_most_similar passed.
[+] All unit tests passed successfully!
```

### 3.3. Chạy thực nghiệm trên Notebook (`experiments.ipynb`)
Mở notebook bằng Jupyter Lab / Jupyter Notebook hoặc VS Code:
```bash
jupyter notebook lab03/experiments.ipynb
```
Notebook tích hợp đầy đủ các phần:
- **Mục 1 & 2:** Kiểm chứng ma trận đồng xuất hiện trên Toy Corpus (Window = 1, 2, 5).
- **Mục 3:** Thực nghiệm 1 trên ngữ liệu mẫu (Đo độ thưa ma trận và số phần tử khác không).
- **Mục 4 & 5:** Thực nghiệm 2 Word2Vec (10.000 documents, 214.188 câu), khảo sát 7 từ mục tiêu và vẽ đồ thị phân cụm PCA 2D.
- **Mục 6:** Thực nghiệm 3 khảo sát Context Window ($w \in \{2, 5, 10\}$).
- **Mục 7:** Thực nghiệm 4 khảo sát Embedding Dimension ($d \in \{50, 100, 300\}$).
- **Mục 8 & 9:** Đánh giá độ tương đồng từ vựng và suy luận tương đồng ($king - man + woman = queen$).
- **Mục 10:** Ứng dụng Semantic Search với câu truy vấn `medical treatment`.
- **Mục 11:** Xuất bảng số liệu tổng hợp ra file `results.csv`.

---

## 4. Tóm tắt kết quả thực nghiệm chính

### 4.1. Thực nghiệm 1: Ma trận Word-Context và độ thưa
- Kích thước từ vựng ($|V|$): **3.043** từ (Tổng số phần tử ma trận: 9.259.849 ô).
- Khi tăng cửa sổ từ 1 lên 5, số phần tử khác không tăng gấp 4 lần (từ 35.370 lên 149.628), nhưng độ thưa vẫn duy trì trên **98.38%**.

### 4.2. Thực nghiệm 2: Huấn luyện Word2Vec và khảo sát 7 từ mục tiêu
- Dữ liệu: 10.000 documents trích xuất thành 214.188 câu và 3.663.973 tokens.
- Cấu hình: CBOW, vector_size = 100, window = 5, min_count = 2, epochs = 10.
- Từ vựng học được: **53.812** từ. Thời gian huấn luyện: ~16.9 giây.
- Kết quả Top-1 tương đồng nhất:
  - `doctor` $\to$ `physician` (0.7328)
  - `hospital` $\to$ `clinic` (0.7036)
  - `patient` $\to$ `diagnosis` (0.6368)
  - `disease` $\to$ `chronic` (0.8523)
  - `computer` $\to$ `desktop` (0.7696)
  - `football` $\to$ `replica` (0.7914)
  - `banana` $\to$ `pebble` (0.8145)

### 4.3. Bảng số liệu định lượng tổng hợp (`results.csv`)

| Hạng mục | Đối tượng khảo sát | Cấu hình 1 | Cấu hình 2 | Cấu hình 3 |
| :--- | :--- | :--- | :--- | :--- |
| **Context Window** | doctor - physician | w=2 (0.7308) | w=5 (0.7293) | w=10 (0.7343) |
| **Context Window** | doctor - hospital | w=2 (0.4768) | w=5 (0.4219) | w=10 (0.4767) |
| **Context Window** | cat - dog | w=2 (0.7019) | w=5 (0.6516) | w=10 (0.6536) |
| **Dimension** | dim=50 | Thời gian=12.8s | RAM=46.19MB | Analogy=queen |
| **Dimension** | dim=100 | Thời gian=16.9s | RAM=66.71MB | Analogy=queen |
| **Dimension** | dim=300 | Thời gian=32.9s | RAM=148.83MB | Analogy=queen |
| **Word Similarity** | king - queen | Điểm=0.7673 | Model=CBOW-100d | Window=5 |
| **Word Similarity** | doctor - physician | Điểm=0.7293 | Model=CBOW-100d | Window=5 |
| **Word Similarity** | cat - dog | Điểm=0.6516 | Model=CBOW-100d | Window=5 |
| **Word Similarity** | car - automobile | Điểm=0.2919 | Model=CBOW-100d | Window=5 |
| **Word Similarity** | computer - banana | Điểm=0.0682 | Model=CBOW-100d | Window=5 |

### 4.4. Ứng dụng Semantic Search
Với câu truy vấn `medical treatment`, hệ thống truy xuất chính xác tài liệu nghiên cứu y khoa về bệnh nhân ung thư và phẫu thuật y tế ở vị trí Top 1 (độ tương đồng 0.7264) dù câu chữ trong văn bản không chứa đúng từ khóa truy vấn.
