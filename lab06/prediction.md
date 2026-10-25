# Prediction trước experiment - Lab 06

Các dự đoán dưới đây được ghi lại trước khi chạy code. Phần đối chiếu ở cuối mỗi mục được bổ sung sau khi đã có kết quả trong experiments.ipynb và calculations.ipynb.

---

## Prediction 1

Câu hỏi: Nếu một key có dot product lớn hơn các key khác, attention weight của nó sẽ như thế nào?

Prediction: Key đó sẽ nhận attention weight lớn nhất, và phần chênh lệch weight sẽ lớn hơn phần chênh lệch score.

Reason: Softmax dùng hàm mũ nên giữ nguyên thứ tự của score, score cao hơn thì weight cao hơn. Ngoài ra hàm mũ khuếch đại khoảng cách, hai score hơn nhau 1 đơn vị thì weight hơn nhau khoảng 2.7 lần.

Confidence: 95%

Đối chiếu: Đúng. Ở bài 2, score [1, 2, 3] cho weight khoảng [0.09, 0.245, 0.665]. Ở Experiment 1, từ good có score cao nhất 0.90 và cũng nhận weight cao nhất khoảng 19.7%.

---

## Prediction 2

Câu hỏi: Nếu tất cả scores giống nhau, s = [1, 1, 1, 1], attention weights sẽ như thế nào?

Prediction: Weights chia đều, mỗi phần tử 0.25.

Reason: Các exp(s_i) bằng nhau nên mỗi phần tử chiếm 1/4 tổng. Giá trị tuyệt đối của score không quan trọng, chỉ khoảng cách giữa các score mới quyết định phân phối. Khi đó context vector chỉ còn là trung bình cộng của các value, attention không còn chọn lọc gì nữa.

Confidence: 99%

Đối chiếu: Đúng. Đây cũng là lý do ở phần numerical stability, trừ đi max(x) không làm thay đổi kết quả. Trong biểu đồ temperature, khi T tăng lên thì entropy tiến dần tới log(10), đúng bằng entropy của phân phối đều trên 10 token.

---

## Prediction 3

Câu hỏi: Nếu một score lớn hơn rất nhiều, s = [1, 1, 10], attention sẽ tập trung như thế nào?

Prediction: Gần như toàn bộ trọng số dồn vào phần tử thứ ba, khoảng 99.98%, hai phần tử còn lại chỉ còn khoảng 0.01% mỗi cái.

Reason: exp(10) lớn hơn exp(1) khoảng e mũ 9, tức hơn 8000 lần. Softmax lúc này gần giống argmax. Mình đoán thêm là trường hợp này không tốt cho việc học, vì gradient đi vào hai phần tử còn lại gần như bằng 0.

Confidence: 90%

Đối chiếu: Đúng về xu hướng. Bảng temperature cho thấy chỉ cần score [4, 2, 0] với T bằng 0.5 (tương đương [8, 4, 0]) là max weight đã lên 0.98. Bảng d_k trong phần scaling cũng cho thấy khi không chia căn d_k, với d_k bằng 1024 thì max weight trung bình gần bằng 1, softmax gần như chọn cứng một key.

---

## Prediction 4

Câu hỏi: Nếu sequence tăng từ 10 tokens lên 1000 tokens, chi phí tính attention thay đổi thế nào?

Prediction: Số phần tử của ma trận attention tăng 10000 lần, từ 100 lên 1 triệu. Thời gian chạy thực tế cũng tăng mạnh nhưng có thể không đúng 10000 lần vì với T nhỏ thì chi phí cố định của NumPy chiếm phần lớn.

Reason: Ma trận attention có kích thước T x T nên tăng theo bình phương của T. T tăng 100 lần thì số phần tử tăng 100 bình phương lần.

Confidence: 85%

Đối chiếu: Đúng về số phần tử. Về thời gian thì đúng như dự đoán là không tăng đủ 10000 lần ở khoảng T nhỏ, vì với T bằng 10 thời gian gần như chỉ là overhead. Từ khoảng T bằng 1000 trở lên thì thời gian bắt đầu tăng gần theo bình phương. Số liệu cụ thể nằm trong bảng Experiment 5.

---

## Prediction 5

Câu hỏi: Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?

Prediction: Không.

Reason: Attention weight chỉ được tối ưu gián tiếp để giảm loss phân loại, không có ai dạy mô hình phải nhìn vào từ nào. Mô hình có thể dồn weight vào những từ thường đi kèm với nhãn trong tập train, kể cả khi từ đó không mang cảm xúc. Thêm nữa, weight cao chưa chắc đóng góp nhiều vào output vì còn phụ thuộc vào value và lớp classifier phía sau.

Confidence: 80%

Đối chiếu: Đúng. Trong phần Application và Error analysis có những câu mà mô hình dồn weight vào từ cảm xúc rất hợp lý, nhưng cũng có những review bị đoán sai dù trọng số vẫn rơi vào các từ trông có vẻ quan trọng. Chi tiết nằm trong error_analysis.md.
