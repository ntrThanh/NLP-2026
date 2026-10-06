# Dự đoán trước thực nghiệm (Predictions)

Tài liệu này ghi lại 4 dự đoán khoa học trước khi tiến hành thực nghiệm trên mô hình, tuân thủ Mục 9 của đề bài W3.pdf.

---

## Dự đoán 1: Mức độ tương đồng giữa các từ mục tiêu
- Câu hỏi: Trong các từ doctor, physician, hospital, banana, car, những từ nào gần nhau nhất?
- Prediction: doctor và physician sẽ có độ tương đồng cao nhất trong nhóm, tiếp theo là cặp doctor và hospital. Các từ banana và car sẽ có độ tương đồng gần như bằng 0 với doctor.
- Reason: doctor và physician là cặp từ đồng nghĩa trực tiếp, có thể thay thế cho nhau trong hầu hết các cấu trúc câu khám chữa bệnh. hospital thuộc cùng trường nghĩa y tế với doctor nhưng không mang tính thay thế trực tiếp. car và banana thuộc các miền khái niệm hoàn toàn khác biệt.
- Confidence: 95%

---

## Dự đoán 2: Ảnh hưởng khi tăng kích thước cửa sổ ngữ cảnh
- Câu hỏi: Nếu kích thước context window tăng từ 2 lên 5, độ tương đồng giữa các từ có thay đổi không?
- Prediction: Độ tương đồng giữa các từ sẽ có sự thay đổi rõ rệt. Cụ thể, các cặp từ đồng nghĩa cú pháp chặt chẽ có thể giảm nhẹ hoặc giữ nguyên điểm số, trong khi các từ liên quan theo chủ đề rộng (như doctor và hospital) sẽ tăng điểm tương đồng.
- Reason: Cửa sổ bằng 2 chỉ nắm bắt các quan hệ cú pháp liền kề. Khi mở rộng cửa sổ lên 5, mô hình bao quát được thêm các từ cùng đoạn văn, giúp học được mối liên hệ chủ đề nhưng cũng đồng thời đưa thêm nhiễu từ các hư từ xuất hiện xung quanh.
- Confidence: 90%

---

## Dự đoán 3: Ảnh hưởng khi tăng số chiều vector
- Câu hỏi: Nếu embedding dimension tăng từ 50 lên 100 rồi lên 300, chất lượng mô hình có chắc chắn tăng không?
- Prediction: Chất lượng không chắc chắn tăng liên tục. Tăng từ 50 lên 100 chiều sẽ cải thiện rõ rệt năng lực biểu diễn, nhưng tăng từ 100 lên 300 chiều trên một tập dữ liệu vừa phải sẽ không tạo ra bước nhảy vọt đáng kể mà chủ yếu làm tăng thời gian tính toán và bộ nhớ.
- Reason: Số chiều biểu diễn cần tương xứng với quy mô dữ liệu. Tập dữ liệu 10.000 văn bản không đủ phong phú để lấp đầy không gian 300 chiều, dẫn đến hiện tượng bão hòa hoặc lãng phí tài nguyên tính toán.
- Confidence: 85%

---

## Dự đoán 4: Khả năng học tương đồng trên ngữ liệu nhỏ
- Câu hỏi: Hai từ doctor và physician có chắc chắn gần nhau không nếu tập ngữ liệu chỉ có 100 câu?
- Prediction: Không chắc chắn gần nhau, độ tương đồng có thể rất thấp hoặc hoàn toàn bằng 0.
- Reason: Trên tập dữ liệu chỉ có 100 câu, xác suất để cả doctor và physician cùng xuất hiện là rất thấp. Nếu chúng chỉ xuất hiện một vài lần và không có chung bất kỳ từ ngữ cảnh nào, mô hình đồng xuất hiện hoặc Word2Vec sẽ không thể học được mối liên hệ giữa hai từ này.
- Confidence: 95%
