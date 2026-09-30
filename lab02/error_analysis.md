# Error Analysis — Lab 02

Tài liệu này thực hiện phân tích định tính và định lượng các trường hợp dự đoán từ tiếp theo (**Next-Word Prediction**) của mô hình N-gram Language Model đã được thực hiện trong [experiments.ipynb](file:///media/trong-thanh/Data2/University/Fourth%20Year/NLP/lab02/experiments.ipynb) (Mục 22 của đề bài).

---

## 1. Bảng tổng hợp các ca phân tích

| Loại | Ngữ cảnh (Context) | Mô hình dự đoán | Từ mong đợi (Expected) | Xác suất $P$ | Nguyên nhân xác định |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Đúng (Case 1)** | `natural language` | `processing` | `processing` | $3.7042 \times 10^{-5}$ | Strong domain collocation (Thuật ngữ chuyên ngành) |
| **Đúng (Case 2)** | `one of` | `the` | `the` | $2.2247 \times 10^{-2}$ | Frequent grammatical phrase (Ngữ pháp cố định) |
| **Sai (Case 3)** | `the cat` | `s` | `sat` | $3.7042 \times 10^{-5}$ | Preprocessing, Domain mismatch, Sparsity |
| **Sai (Case 4)** | `machine learning` | `and` | `models` | $6.7214 \times 10^{-5}$ | Context quá ngắn, Stopword tần suất cao |

---

## 2. Chi tiết từng trường hợp

### Trường hợp 1: Dự đoán đúng — Collocation mạnh trong miền chuyên ngành
- **Context:** `natural language`
- **Model prediction:** `processing`
- **Expected:** `processing`
- **Probability:** $3.7042 \times 10^{-5}$
- **Phân tích nguyên nhân:**
  - Cụm từ `natural language processing` là một danh từ ghép cố định (collocation) xuất hiện với tần suất cực kỳ cao trong tập dữ liệu khoa học máy tính / AI.
  - Ngữ cảnh 2 từ trước đó mang tính thông tin định hướng rất mạnh, thu hẹp không gian phân phối từ tiếp theo hầu như chỉ tập trung vào `processing`.

---

### Trường hợp 2: Dự đoán đúng — Cụm từ ngữ pháp phổ biến
- **Context:** `one of`
- **Model prediction:** `the`
- **Expected:** `the`
- **Probability:** $2.2247 \times 10^{-2}$ (xác suất rất cao, chiếm hơn $2.2\%$)
- **Phân tích nguyên nhân:**
  - Cụm `one of the` là một khuôn mẫu ngữ pháp chuẩn và phổ biến bậc nhất trong văn phong học thuật tiếng Anh.
  - Sau `one of`, mạo từ `the` xuất hiện áp đảo so với bất kỳ từ loại nào khác, giúp mô hình n-gram dễ dàng chọn đúng từ tiếp theo.

---

### Trường hợp 3: Dự đoán sai — Do Preprocessing và Lệch miền dữ liệu (Domain Mismatch)
- **Context:** `the cat`
- **Model prediction:** `s`
- **Expected:** `sat` (hoặc `eats`, `is`)
- **Probability:** $3.7042 \times 10^{-5}$
- **Phân tích nguyên nhân:**
  1. **Preprocessing:** Bộ tách từ dựa trên biểu thức chính quy tách các dạng sở hữu cách như `cat's` thành hai token rời rạc `cat` và `s`.
  2. **Domain mismatch:** Tập dữ liệu 30.000 bài báo khoa học kỹ thuật gần như không bao giờ đề cập đến vật nuôi ("cat"). Cụm từ `the cat` chỉ tình cờ xuất hiện trong một vài cụm từ viết tắt hoặc tên biến/mã code.
  3. **Data Sparsity:** Do không có dữ liệu thực sự về `the cat sat`, tần suất của các từ thông thường bằng 0, khiến token rời `s` (từ sở hữu cách) tình cờ chiếm xác suất cao nhất.

---

### Trường hợp 4: Dự đoán sai — Context quá ngắn và nhiễu Stopword
- **Context:** `machine learning`
- **Model prediction:** `and`
- **Expected:** `models` (hoặc `algorithms`, `techniques`)
- **Probability:** $6.7214 \times 10^{-5}$
- **Phân tích nguyên nhân:**
  1. **Context quá ngắn (Markov 2 từ):** Chỉ nhìn 2 từ `machine learning` không đủ để xác định cấu trúc câu là danh từ ghép bổ nghĩa (`machine learning models`) hay liệt kê liên từ (`machine learning and data science`).
  2. **Hiệu ứng Stopword:** Từ nối `and` xuất hiện ở khắp mọi nơi trong văn bản, có tần suất unigram và bigram vượt trội, lấn át các danh từ kỹ thuật cụ thể.
  3. **Hạn chế của Laplace Smoothing:** Việc cộng thêm 1 cho tất cả các từ trong từ vựng khổng lồ ($|V| \approx 186K$) làm phẳng phân phối quá mức, khiến các từ có tần suất nền cao như `and` dễ dàng vượt lên.

---

## 3. Bài học rút ra & Hướng giải quyết

1. **Vấn đề Tokenization / Preprocessing:**
   - Cần xử lý dấu nháy đơn, chữ viết tắt và sở hữu cách một cách nhất quán (ví dụ giữ nguyên `cat's` hoặc chuẩn hóa theo bộ tokenizer chuyên dụng như Byte-Pair Encoding / WordPiece).

2. **Vấn đề Smoothing:**
   - Laplace (Add-1) smoothing quá "thô bạo" khi $|V|$ lớn, làm loãng xác suất của các n-gram có ý nghĩa. Cần chuyển sang các kỹ thuật làm mịn hiện đại hơn như **Absolute Discounting**, **Jelinek-Mercer Interpolation**, hoặc tối ưu nhất là **Kneser-Ney Smoothing**.

3. **Giới hạn cố hữu của N-gram:**
   - N-gram hoàn toàn bất lực trước hiện tượng đồng nghĩa (Synonyms) và phụ thuộc xa (Long-range dependencies). Đây chính là động lực chuyển dịch tất yếu sang **Neural Language Models** (RNN, LSTM, Transformer/GPT) - nơi ngữ cảnh được biểu diễn bằng vector liên tục (dense embeddings) và cơ chế Attention có thể nhìn toàn bộ câu.
