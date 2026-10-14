# Reflection & Individual Learning Check — Lab 04

---

## 1. Mạch nối kiến thức: Từ LAB 01 đến LAB 04 (Mục 29)

Curriculum của môn học được thiết kế với tính logic và liên kết rất chặt chẽ:

```text
LAB 01: Raw Text -> TF-IDF -> Sparse Representation -> Information Retrieval / Search
   ↓
LAB 02: Raw Text -> N-gram Counts -> Conditional Probability -> Language Model / Perplexity
   ↓
LAB 03: Raw Text -> Local Context Window -> Dense Word Embeddings -> Semantic Similarity
   ↓
LAB 04: Raw Text -> Representation (TF-IDF vs Word2Vec) -> Classifier -> Evaluation & Error Analysis
```

### Giới hạn cốt lõi phát hiện trong Lab 04:
1. **$\text{TF-IDF} \neq \text{Semantic Understanding}$:**
   - TF-IDF hoàn toàn dựa trên việc khớp từ vựng mặt chữ (lexical matching). Hai từ đồng nghĩa như `excellent` và `outstanding` có hai chiều vector hoàn toàn trực giao, mô hình không biết chúng mang cùng một sắc thái nếu một trong hai từ chưa từng xuất hiện đủ nhiều trong tập train.
   - TF-IDF không có khả năng hiểu ngữ cảnh tổng thể của câu.
2. **$\text{Word2Vec} \neq \text{Contextual Understanding}$:**
   - Mỗi từ trong Word2Vec tĩnh chỉ có **duy nhất một vector cố định** (static embedding). Nó không thể phân biệt được từ đa nghĩa (*Polysemy*) như:
     - `"bank"` = tổ chức tài chính
     - `"bank"` = bờ sông
   - Khi áp dụng vào bài toán phân loại văn bản bằng cách tính vector trung bình ($\frac{1}{n} \sum \vec{w}_i$), toàn bộ trật tự từ bị xóa sạch:
     - `"this movie is good"`
     - `"this movie is not good"`
     cả hai câu đều có vector gần như trùng khớp nhau ($\text{cosine} = 0.9361$).
3. **Cầu nối sang LAB 05 (Sequence Models — RNN / LSTM / GRU):**
   - Sự thất bại của Bag-of-Words và Averaged Embeddings trước hiện tượng trật tự từ và phủ định chính là động lực trực tiếp để bước sang **Lab 05**.
   - Ở các lab tiếp theo, các mô hình dạng chuỗi (Recurrent Neural Networks) và sau đó là Transformer (Self-Attention) sẽ cho phép mô hình hóa trực tiếp sự biến đổi trạng thái theo thời gian của câu, hiểu được phạm vi ảnh hưởng của từ `not` lên các từ đứng sau.

---

## 2. Individual Learning Check — Chuẩn bị vấn đáp 3 phút (Mục 25)

Dưới đây là câu trả lời chuẩn xác, súc tích cho 6 câu hỏi vấn đáp ngẫu nhiên của giảng viên:

### Câu 1: Tại sao Accuracy 95% chưa chắc đã tốt?
- **Trả lời:** Accuracy chỉ đo tỷ lệ đoán đúng trên tổng số mẫu mà không quan tâm đến phân phối giữa các lớp. Trong tập dữ liệu mất cân bằng nghiêm trọng (ví dụ 950 mẫu tiêu cực, 50 mẫu tích cực), một mô hình "ngớ ngẩn" luôn luôn dự đoán mọi mẫu là Negative cũng sẽ đạt Accuracy đúng 95%. Tuy nhiên, mô hình này hoàn toàn vô dụng vì không phát hiện được bất kỳ một ca Positive nào ($\text{Recall} = 0, \text{F1} = 0$). Để đánh giá đúng, bắt buộc phải xem xét thêm **Precision**, **Recall** và **F1-score**.

### Câu 2: TF-IDF khác Average Word2Vec ở những điểm cốt lõi nào?
- **Trả lời:**
  - **Không gian biểu diễn:** TF-IDF tạo ra vector **thưa (sparse)** với số chiều rất lớn bằng kích thước từ điển ($|V| \approx 87{,}000$), phần lớn là số 0. Ngược lại, Word2Vec tạo ra vector **đặc (dense)** với số chiều cố định và nhỏ gọn ($d = 100$ hoặc $300$).
  - **Bản chất thông tin:** TF-IDF mang tính **từ vựng (lexical)**, giữ nguyên thông tin về sự xuất hiện của từng từ cụ thể. Word2Vec mang tính **ngữ nghĩa (semantic)**, các từ đồng nghĩa sẽ nằm gần nhau trong không gian vector.
  - **Hành vi trong phân loại:** TF-IDF giữ được sức nặng của các từ khóa cảm xúc đặc trưng cực mạnh (*terrible, wonderful*), trong khi Average Word2Vec làm loãng các từ khóa này do bị cộng dồn và chia trung bình với hàng trăm từ trung tính trong văn bản dài.

### Câu 3: Tại sao câu *"this movie is not good"* có thể gây lỗi cho mô hình?
- **Trả lời:** Cả mô hình Bag-of-Words (Unigram) và Average Word2Vec đều **mất trật tự từ (loss of word order)**. Từ `not` chỉ được xem như một đặc trưng độc lập được cộng vào vector giống như các từ khác. Không có cơ chế nào để từ `not` liên kết và đảo ngược cực tính của tính từ `good` đứng liền sau. Vì `good` có trọng số dương rất lớn trong tập train, tổng logit vẫn nghiêng về phía tích cực, dẫn đến việc mô hình dự đoán nhầm thành Positive.

### Câu 4: Data Leakage xảy ra ở đâu nếu fit TF-IDF trước khi chia Train / Validation / Test?
- **Trả lời:** Rò rỉ dữ liệu xảy ra vì hai lý do:
  1. **Từ điển (Vocabulary Leakage):** Các từ hiếm chỉ có trong tập Test đã bị đưa vào bảng từ vựng của mô hình ngay từ đầu.
  2. **Trọng số IDF (Frequency Leakage):** Công thức $\text{IDF}(t) = \log \frac{N}{\text{DF}(t)}$ đã sử dụng tần số tài liệu của toàn bộ tập Test để tính toán trọng số.
  Điều này khiến mô hình đã gián tiếp "nhìn thấy trước" thông tin thống kê của tập Test, làm kết quả đánh giá thực nghiệm bị lạc quan giả tạo (over-optimistic) và không phản ánh đúng năng lực tổng quát hóa ngoài thực tế.

### Câu 5: Precision và Recall khác nhau thế nào? Khi nào cần ưu tiên chỉ số nào?
- **Trả lời:**
  - **Precision:** $\frac{TP}{TP + FP}$ — Trong số những mẫu mô hình *dự đoán là dương tính*, có bao nhiêu mẫu *thực sự đúng*. Đo lường mức độ tin cậy, tránh báo động giả (False Positive).
  - **Recall:** $\frac{TP}{TP + FN}$ — Trong số những mẫu *thực sự là dương tính*, mô hình *bắt trúng được bao nhiêu*. Đo lường khả năng bao phủ, tránh bỏ sót (False Negative).
  - **Ví dụ ưu tiên:** Trong phát hiện Email Spam, ta ưu tiên **Precision** (không được ném nhầm thư công việc quan trọng của người dùng vào thùng rác - tránh FP). Trong chẩn đoán bệnh ung thư hoặc phát hiện mã độc, ta ưu tiên **Recall** (thà xét nghiệm lại nhầm còn hơn bỏ sót người bệnh - tránh FN).

### Câu 6: Tại sao thêm Bigram có thể cải thiện sentiment classification?
- **Trả lời:** Bigram kết hợp 2 từ liền kề thành một đơn vị đặc trưng duy nhất. Điều này giúp mô hình nắm bắt được **ngữ cảnh cục bộ (local context)** và **cấu trúc phủ định trực tiếp** (ví dụ cụm `"not good"` trở thành một đặc trưng riêng có trọng số âm, hoàn toàn tách biệt với từ đơn `"good"` mang trọng số dương). Nhờ đó, hiện tượng nhầm lẫn câu phủ định giảm đi đáng kể.

---

## 3. AI Assistance Statement (Mục 24)

- **Tool:** Google Antigravity AI Assistant (mô hình tiên tiến của Google DeepMind).
- **Purpose:** 
  - Hỗ trợ xây dựng cấu trúc pipeline theo đúng chuẩn Scikit-learn (ngăn ngừa Data Leakage).
  - Tối ưu hóa hàm tiền xử lý văn bản bằng biểu thức chính quy (Regex) và giải mã HTML entities.
  - Hỗ trợ vẽ và xuất biểu đồ Confusion Matrix trực quan.
- **What was generated:** Mã nguồn khung thực nghiệm trong `preprocessing.py`, các cell tính toán số liệu trong `experiments.ipynb`, và định dạng bảng biểu Markdown.
- **What was modified:** Đã tự tay tinh chỉnh siêu tham số mô hình (`min_df=2`, `ngram_range=(1, 2)`), đối soát trực tiếp các bài review bị phân loại sai trên tập Test để viết báo cáo Error Analysis, và kiểm tra tính hợp lệ của toàn bộ ma trận dữ liệu.
- **How the result was verified:** Toàn bộ thực nghiệm được chạy độc lập từ đầu đến cuối trên môi trường Python 3.12, xuất ra file `results.csv` và ảnh `confusion_matrix.png`, kiểm tra không có bất kỳ rò rỉ dữ liệu nào giữa các tập Train/Val/Test.

