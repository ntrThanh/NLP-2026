# Prediction Before Experiment — Lab 02

> **Ghi chú:** Các dự đoán dưới đây được lập ra trước khi chạy thực nghiệm trên tập ngữ liệu 30.000 documents (đã được ghi chép trong bản scan viết tay [NguyenTrongThanh_calculations.pdf](file:///media/trong-thanh/Data2/University/Fourth%20Year/NLP/lab02/NguyenTrongThanh_calculations.pdf)). Phần "Kiểm chứng sau thực nghiệm" được đối chiếu từ kết quả chạy notebook [experiments.ipynb](file:///media/trong-thanh/Data2/University/Fourth%20Year/NLP/lab02/experiments.ipynb).

---

## 1. Prediction 1: Vocabulary Size khi tăng bậc $n$

- **Prediction:** Khi chuyển từ Unigram $\to$ Bigram $\to$ Trigram, kích thước từ vựng (**Vocabulary Size** $|V|$) **KHÔNG THAY ĐỔI**.
- **Reason:** Vocabulary được định nghĩa là tập hợp các từ đơn độc nhất (unique words / tokens) cấu thành nên ngôn ngữ của corpus. Dù mô hình hóa ngữ cảnh bằng các cặp từ hay bộ ba từ, không gian từ vựng cơ sở vẫn giữ nguyên là $|V|$.
- **Confidence:** Cao ($95\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**. 
  Trên tập corpus 30.000 tài liệu, kích thước từ vựng duy trì cố định là $|V| = 186.618$ từ trên cả 3 mô hình Unigram, Bigram và Trigram.

---

## 2. Prediction 2: Số lượng unique $n$-grams

- **Prediction:** Số lượng unique $n$-gram sẽ **TĂNG RẤT MẠNH** theo cấp số cộng hoặc tiệm cận số token khi tăng $n$ ($N_{\text{unigram}} \ll N_{\text{bigram}} \ll N_{\text{trigram}}$).
- **Reason:** Không gian tổ hợp chuỗi $n$ từ tăng theo quy mô $|V|^n$. Trong thực tế, các cách kết hợp từ trong văn bản tự nhiên rất phong phú, khiến số lượng cụm 2 từ và 3 từ khác nhau bùng nổ theo chiều dài văn bản.
- **Confidence:** Rất cao ($99\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Unique Unigrams: **186.618**
  - Unique Bigrams: **2.739.018** (gấp $\approx 14.7$ lần)
  - Unique Trigrams: **6.404.589** (gấp $\approx 34.3$ lần)

---

## 3. Prediction 3: Nguy cơ gặp Zero Probability

- **Prediction:** Mô hình **Trigram** có nguy cơ gặp **Zero Probability cao nhất**, tiếp theo là Bigram, và Unigram có xác suất gặp zero thấp nhất.
- **Reason:** Số lượng tổ hợp có thể có của trigram là $|V|^3$ (cực kỳ khổng lồ, cỡ $10^{15}$). Bất kỳ tập ngữ liệu hữu hạn nào cũng chỉ chứa một lát cắt rất nhỏ của không gian này. Do đó, khi gặp câu mới, xác suất một cụm 3 từ chưa từng xuất hiện trong tập train là cực lớn $\implies C(w_{t-2}, w_{t-1}, w_t) = 0$.
- **Confidence:** Rất cao ($95\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Tỷ lệ n-gram chỉ xuất hiện 1 lần (singletons) của Trigram lên tới **86.21%** (so với 71.30% của Bigram và 47.41% của Unigram).
  - Khi đánh giá mô hình MLE trên tập Validation/Test, cả Bigram MLE và Trigram MLE đều nhận **PPL = $\infty$** do gặp zero probability ngay lập tức.

---

## 4. Prediction 4: Perplexity trên Training Set

- **Prediction:** Mô hình **Trigram MLE** sẽ có **Perplexity THẤP NHẤT** trên Training Set.
- **Reason:** Trigram có nhiều tham số tự do nhất và điều kiện hóa trên ngữ cảnh 2 từ trước đó. Trên chính tập dữ liệu nó được huấn luyện, Trigram có khả năng "học vẹt" (memorize) các cụm từ chính xác nhất, do đó gán xác suất cao nhất cho tập train, dẫn đến perplexity thấp nhất.
- **Confidence:** Cao ($90\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Training PPL của Trigram MLE là **19.71**, vượt trội so với Bigram MLE (**209.81**) và Unigram (**1915.23**).
  - Tuy nhiên, khi áp dụng Laplace smoothing trên tập từ vựng khổng lồ ($|V| = 186.618$), hiệu ứng phạt làm mịn đã đẩy Trigram Laplace lên 27.462,79.

---

## 5. Prediction 5: Trigram vs Bigram trên Corpus Nhỏ

- **Prediction:** Nếu corpus rất nhỏ, Trigram **KHÔNG CHẮC CHẮN TỐT HƠN** Bigram; thậm chí Bigram hoặc Unigram có thể tổng quát hóa tốt hơn.
- **Reason:** Corpus nhỏ không cung cấp đủ số lần xuất hiện cho các cụm 3 từ. Hiện tượng thưa dữ liệu (Data Sparsity) sẽ áp đảo khả năng học ngữ cảnh, khiến hầu hết các phép tính xác suất của Trigram bị rơi vào smoothing và gán giá trị làm mịn nhân tạo, làm giảm tính phân biệt của mô hình.
- **Confidence:** Cao ($85\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Trên tập Test, mô hình Unigram đạt Perplexity tốt nhất (**2050.78**), tiếp đến là Bigram (**6703.09**), còn Trigram có Perplexity tệ nhất (**43047.01**) do bị phạt nặng nề bởi $|V| = 186.618$ trong mẫu số làm mịn Laplace.
