# LAB 04 — Text Classification: From Text Representation to Supervised NLP

* **Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP 2026)  
* **Trường:** Trường Đại học Khoa học Tự nhiên - ĐHQGHN (VNU-HUS)  
* **Giảng viên / TA:** Phạm Ngọc Hải  
* **Hạn nộp:** 23:59 - 14/10/2026  

---

## 1. Tổng quan bài Lab

Tiếp nối chuỗi kiến thức từ **Lab 01** (biểu diễn thưa TF-IDF), **Lab 02** (mô hình ngôn ngữ thống kê N-gram), và **Lab 03** (biểu diễn vector đặc Word2Vec), **Lab 04** đưa các phương pháp biểu diễn văn bản trên vào một bài toán học có giám sát hoàn chỉnh: **Phân loại văn bản (Text Classification)** trên bộ ngữ liệu lớn **IMDb Movie Reviews (50,000 bài đánh giá)**.

### Mục tiêu học tập cốt lõi:
1. **Thiết lập Pipeline chống rò rỉ dữ liệu (No Data Leakage):** Loại bỏ trùng lặp nội bộ và phân chia tập dữ liệu nghiêm ngặt theo tỷ lệ 70% Train, 15% Validation, 15% Test. Quá trình học từ vựng và IDF chỉ được `fit` trên tập Train.
2. **So sánh các phương pháp biểu diễn:** Đặt lên bàn cân giữa biểu diễn thưa/từ vựng (**TF-IDF**) và biểu diễn đặc/ngữ nghĩa (**Average Word2Vec**) cùng các bộ phân loại tuyến tính (**Logistic Regression**, **Linear SVM**).
3. **Kiểm chứng giới hạn của Bag-of-Words & Averaged Embeddings:** Thử nghiệm trực tiếp trên các cấu trúc trật tự từ và phủ định phức tạp (*"this movie is not good"*).
4. **Mở rộng ngữ cảnh với N-gram:** Khảo sát sự đánh đổi giữa kích thước không gian đặc trưng và hiệu năng khi chuyển từ Unigram sang Unigram + Bigram.
5. **Giải thích mô hình và phân tích lỗi:** Khảo sát trọng số của các từ khóa quyết định (Model Interpretability), vẽ biểu đồ Confusion Matrix và phân tích định tính các ca phân loại sai trên tập Test.

---

## 2. Cấu trúc thư mục nộp bài (Deliverables Mục 26)

Thư mục `lab04/` tuân thủ nghiêm ngặt quy định tại Mục 26 của đề bài `W4.pdf`:

```text
lab04/
├── README.md               # Báo cáo tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md         # Lời giải chi tiết 4 bài tập tính toán lý thuyết (Mục 6-9)
├── prediction.md           # 5 dự đoán khoa học trước thực nghiệm kèm đối chứng (Mục 10)
├── preprocessing.py        # Module độc lập xử lý HTML, chuẩn hóa và tách token
├── experiments.ipynb       # Toàn bộ mã nguồn thực nghiệm và biểu đồ có output đầy đủ
├── results.csv             # Bảng số liệu định lượng tổng hợp tất cả các mô hình
├── confusion_matrix.png    # Đồ thị ma trận nhầm lẫn chuẩn xuất từ tập Test
├── error_analysis.md       # Báo cáo phân tích định tính 6 bài review thực tế (Mục 21)
├── reflection.md           # Phản tư kiến thức, AI statement và câu hỏi vấn đáp (Mục 25 & 29)
└── W4.pdf                  # Đề bài gốc của môn học
```

---

## 3. Hướng dẫn chạy và tái hiện kết quả

### 3.1. Thiết lập môi trường
Yêu cầu môi trường Python 3.10+ với các thư viện:
```bash
pip install numpy pandas scikit-learn matplotlib gensim
```

### 3.2. Chạy kiểm thử module tiền xử lý (`preprocessing.py`)
Kiểm tra tính đúng đắn của các hàm làm sạch HTML, chuẩn hóa ký tự và tokenize:
```bash
python3 lab04/preprocessing.py
```

### 3.3. Chạy toàn bộ thực nghiệm trên Jupyter Notebook
Mở notebook bằng Jupyter Lab hoặc VS Code:
```bash
jupyter notebook lab04/experiments.ipynb
```
Notebook tích hợp tuần tự các bước:
- **Experiment 1 (Mục 11):** Khảo sát phân phối nhãn, dữ liệu thiếu, trùng lặp và phân phối độ dài văn bản.
- **Mục 12:** Lọc 418 mẫu trùng lặp và lưu trữ các tập dữ liệu tại `datasets/splits/`.
- **Experiment 2 (Mục 13 & 14):** Huấn luyện mô hình Baseline TF-IDF (Unigram) + Logistic Regression.
- **Experiment 3 (Mục 15 & 16):** Huấn luyện Word2Vec, tính trung bình vector văn bản, so sánh với TF-IDF và Linear SVM.
- **Experiment 4 (Mục 17):** Thí nghiệm kiểm chứng thứ tự từ và phủ định trên hai câu mẫu.
- **Experiment 5 (Mục 18):** Mở rộng TF-IDF Unigram + Bigram (`min_df=2`), đo lường độ phình từ điển và F1-score.
- **Model Interpretability (Mục 22):** Trích xuất Top 10 từ có trọng số dương và âm lớn nhất.
- **Evaluation & Error Analysis (Mục 20 & 21):** Vẽ ma trận nhầm lẫn `confusion_matrix.png` và trích xuất bảng 6 mẫu review kiểm chứng.

---

## 4. Tóm tắt kết quả thực nghiệm chính

### 4.1. Bảng so sánh các phương pháp biểu diễn trên tập Test ([`results.csv`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab04/results.csv))

| Representation | Classifier | Vocabulary ($|V|$) | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (Unigram)** | **Logistic Regression (Baseline)** | 87,734 | 0.9016 | 0.8952 | 0.9105 | **0.9028** |
| **Average Word2Vec** | **Logistic Regression** | 100 dimensions | 0.8428 | 0.8368 | 0.8532 | **0.8449** |
| **TF-IDF (Unigram)** | **Linear SVM** | 87,734 | 0.9060 | 0.9011 | 0.9129 | **0.9070** |
| **TF-IDF (Uni+Bigram)** | **Logistic Regression** | **555,050** | **0.9062** | 0.8992 | 0.9156 | **0.9074** |

### 4.2. Khám phá cốt lõi:
1. **Tại sao TF-IDF tốt hơn Average Word2Vec trên dữ liệu văn bản dài?**
   - Bài đánh giá IMDb trung bình dài hơn 230 từ. Việc lấy trung bình cộng làm "loãng" các từ cảm xúc mạnh bởi hàng trăm từ trung tính. Ngược lại, tính chất thưa (sparse) và từ vựng (lexical) của TF-IDF giữ nguyên sức nặng của các từ cảm xúc đặc trưng.
2. **Hạn chế của Bag-of-Words trước từ phủ định (`not`):**
   - Cả TF-IDF Unigram và Average Word2Vec đều đoán câu *"this movie is not good"* thành **Positive** vì từ `good` lấn át từ `not` và mô hình không có cơ chế thứ tự từ.
   - Thêm Bigram (`"not good"`) đã đảo chiều xác suất thành công từ 63.19% xuống còn **15.59%**, giúp mô hình phân loại chính xác nhãn Negative.
3. **Top đặc trưng định hướng mô hình (Model Weights):**
   - **Top Positive ($w > 0$):** `great` (+8.69), `excellent` (+7.07), `best` (+5.74), `perfect` (+5.32), `wonderful` (+5.00).
   - **Top Negative ($w < 0$):** `worst` (-10.47), `bad` (-8.84), `awful` (-7.96), `waste` (-7.15), `boring` (-6.87).

