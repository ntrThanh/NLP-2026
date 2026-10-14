# Prediction Before Experiment — Lab 04

> **Ghi chú:** Các dự đoán dưới đây được thiết lập trước khi chạy thực nghiệm theo yêu cầu tại Mục 10 của đề bài W4.pdf. Phần "Kiểm chứng sau thực nghiệm" được đối chiếu từ kết quả thực tế thu được trong [experiments.ipynb](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab04/experiments.ipynb) và [results.csv](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab04/results.csv).

---

## 1. Prediction 1: TF-IDF + Logistic Regression có hoạt động tốt không?

- **Prediction:** **CÓ**, mô hình TF-IDF + Logistic Regression sẽ hoạt động rất tốt trên bài toán phân loại cảm xúc IMDb (Accuracy và F1 dự kiến đạt từ 88% đến 90%+).
- **Reason:** 
  - Đánh giá cảm xúc trong văn bản (Sentiment Analysis) phụ thuộc rất lớn vào các từ khóa mang sắc thái cảm xúc cực tính mạnh (vd: *excellent, wonderful, terrible, awful, waste*).
  - TF-IDF gán trọng số cao cho các từ đặc trưng xuất hiện thường xuyên trong từng tài liệu nhưng hiếm xuất hiện trên toàn bộ corpus, giúp làm nổi bật các từ mang tính phân loại cao.
  - Logistic Regression là mô hình tuyến tính tối ưu hóa siêu phẳng phân chia rất hiệu quả trên không gian vector thưa có số chiều lớn ($|V| \approx 87{,}000$).
- **Confidence:** Rất cao ($95\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Mô hình Baseline TF-IDF + Logistic Regression đạt **Accuracy = 90.16%** và **F1-score = 0.9028** trên tập Test (7.438 bài đánh giá), thời gian huấn luyện chỉ mất ~8.24 giây.

---

## 2. Prediction 2: Average Word2Vec có chắc chắn tốt hơn TF-IDF không?

- **Prediction:** **KHÔNG**, Average Word2Vec không chắc chắn tốt hơn, thậm chí nhiều khả năng cho kết quả **KÉM HƠN** TF-IDF trên dữ liệu văn bản dài.
- **Reason:** 
  - Mặc dù Word2Vec học được vector dày đặc (dense) mang tính tương đồng ngữ nghĩa, nhưng phương pháp biểu diễn văn bản bằng cách lấy trung bình cộng tất cả vector từ ($\vec{x} = \frac{1}{n} \sum_{i} \vec{w}_i$) sẽ làm "pha loãng" các từ khóa cảm xúc then chốt.
  - Một bài đánh giá IMDb trung bình dài hơn 230 từ, chứa vô số từ trung tính (neutral/stopwords). Khi cộng dồn và chia trung bình, tín hiệu cảm xúc của một vài tính từ mạnh bị suy giảm nghiêm trọng.
  - Ngoài ra, phép tính trung bình xóa bỏ hoàn toàn thứ tự từ (word order) và cấu trúc phủ định (*negation*).
- **Confidence:** Cao ($85\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Mô hình Average Word2Vec + Logistic Regression chỉ đạt **Accuracy = 84.28%** và **F1-score = 0.8449**, thấp hơn gần **6%** so với TF-IDF + Logistic Regression (90.16%).

---

## 3. Prediction 3: Tăng Vocabulary Size có chắc chắn làm tăng Accuracy không?

- **Prediction:** **KHÔNG**, tăng Vocabulary Size không chắc chắn làm tăng Accuracy, thậm chí có thể gây bão hòa hoặc giảm hiệu năng do quá khớp (overfitting).
- **Reason:** 
  - Nếu mở rộng từ vựng vô hạn định bằng cách giữ lại toàn bộ các từ chỉ xuất hiện 1 lần (hapax legomena), từ viết sai chính tả, tên riêng hiếm, hoặc ký tự rác, không gian đặc trưng sẽ bị phình to với phần lớn là nhiễu (noise).
  - Các đặc trưng cực hiếm không mang tính tổng quát cho tập Test nhưng lại khiến mô hình tuyến tính dễ ghi nhớ nhiễu của tập Train, làm tăng chi phí tính toán và bộ nhớ mà không cải thiện độ chính xác thực tế.
- **Confidence:** Rất cao ($90\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Khi mở rộng từ vựng sang Unigram + Bigram không lọc (`min_df=1`), từ vựng tăng vọt lên gần 2 triệu từ ($1{,}929{,}469$), thời gian huấn luyện tăng gấp 5 lần nhưng kết quả không vượt trội so với việc lọc `min_df=2` (555,050 từ vựng cho Accuracy = 90.62%, F1 = 0.9074).

---

## 4. Prediction 4: Loại bỏ Stopwords có chắc chắn cải thiện Performance không?

- **Prediction:** **KHÔNG**, việc loại bỏ Stopwords một cách máy móc không chắc chắn làm tăng performance trong bài toán Sentiment Analysis.
- **Reason:** 
  - Danh sách stopword tiêu chuẩn thường chứa các từ phủ định (*not, no, nor, never, hardly*) và các từ tăng cường sắc thái (*very, too, but, more*).
  - Nếu xóa bỏ các từ này, câu mang nghĩa phủ định như *"this movie is not good"* sẽ bị biến đổi thành *"movie good"*, làm đảo ngược hoàn toàn cực tính cảm xúc của văn bản và khiến mô hình dự đoán sai nghiêm trọng.
- **Confidence:** Cao ($85\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Giữ lại các từ như `not` kết hợp với biểu diễn Bigram (*"not good"*) đã giúp xác suất dự đoán sai của mô hình giảm mạnh từ 63.19% xuống 15.59%, chuyển từ dự đoán nhầm thành dự đoán đúng nhãn Negative.

---

## 5. Prediction 5: Accuracy và F1 có thể cho kết luận khác nhau không?

- **Prediction:** **CÓ**, Accuracy và F1 hoàn toàn có thể cho các kết luận đánh giá khác nhau về chất lượng của mô hình.
- **Reason:** 
  - **Accuracy** đo tỷ lệ số mẫu dự đoán đúng trên toàn bộ tập dữ liệu:
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
    Chỉ số này rất dễ gây hiểu lầm trong các bài toán mất cân bằng lớp (Class Imbalance).
  - **F1-score** là trung bình điều hòa giữa Precision và Recall:
    $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
    F1 chỉ đạt giá trị cao khi cả Precision (độ chuẩn xác khi dự đoán dương tính) và Recall (khả năng bao phủ không bỏ sót ca dương tính) đều tốt.
- **Confidence:** Rất cao ($99\%$).
- **Kiểm chứng sau thực nghiệm:** **ĐÚNG**.
  - Dù tập Test IMDb cân bằng hoàn hảo (50% pos - 50% neg), quan sát chi tiết cho thấy Precision (89.52%) và Recall (91.05%) có sự lệch nhẹ: mô hình nhạy hơn trong việc bắt các review tích cực (ít False Negatives), nhưng tỷ lệ cảnh báo nhầm (False Positives) cao hơn. F1-score phản ánh toàn diện sự đánh đổi này hơn là chỉ nhìn vào một con số Accuracy 90.16%.

