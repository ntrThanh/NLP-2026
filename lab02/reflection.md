# Reflection & Learning Check — Lab 02

---

## 24. Reflection

### Câu 1: Nếu tăng (n), mô hình nhận thêm thông tin gì?

Mô hình biết thêm trật tự của các từ đứng trước và ngữ cảnh gần của câu thay vì chỉ đếm từ riêng lẻ.

### Câu 2: Tại sao tăng (n) lại làm sparsity tăng?

Càng ghép nhiều từ thì càng khó gặp lại cụm từ đó trong tập dữ liệu. Số tổ hợp từ tăng quá nhanh trong khi dữ liệu có hạn nên hầu hết các cụm từ đều có số lần xuất hiện bằng 0.

### Câu 3: Tại sao smoothing cần thiết?

Để tránh xác suất bằng 0. Nếu gặp một từ hoặc cụm từ chưa từng thấy lúc train, xác suất của cả câu sẽ bị nhân về 0. Smoothing giúp chia bớt xác suất cho các từ chưa thấy để mô hình không bị lỗi.

### Câu 4: Perplexity đo điều gì?

Đo độ bối rối của mô hình khi đọc câu mới. Perplexity càng thấp nghĩa là mô hình đoán càng chuẩn và ít bị bất ngờ.

### Câu 5: Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.

Không. Perplexity thấp chỉ là đoán đúng xác suất từng từ trên bề mặt thống kê. Mô hình vẫn có thể lặp từ liên tục hoặc sinh ra câu vô nghĩa mà con người đọc không hiểu.

### Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?

N-gram chỉ biết đếm từ giống nhau chứ không hiểu nghĩa của từ (không biết các từ đồng nghĩa), và chỉ nhìn được 1 đến 2 từ trước đó chứ không nhớ được ngữ cảnh dài của cả đoạn văn.

### Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?

Không. Trigram chỉ nhìn đúng 2 từ đứng ngay trước nó và bỏ qua hoàn toàn 97 từ đầu.

---

## 25. AI Assistance Statement

- Tool: Antigravity AI Assistant
- Purpose: Hỗ trợ tìm kiếm documentation, debug cấu trúc dữ liệu Counter, tối ưu code và kiểm tra cú pháp Python.
- What was generated: Khung hàm xếp hạng câu trong notebook và định dạng bảng thống kê Markdown.
- What was modified: Đã tùy chỉnh lại toàn bộ thuật toán tính xác suất, chuẩn hóa độ dài perplexity và giải thích hiện tượng theo ngữ liệu thực nghiệm.
- How the result was verified: Chạy độc lập các kịch bản kiểm thử trên toy corpus và so sánh giá trị Perplexity thực tế trên tập train/validation/test.

---

## 26. Individual Learning Check (Chuẩn bị vấn đáp)

1. Vì sao bigram có zero probability?
- Vì nhiều cặp từ hợp lý trong thực tế nhưng chưa từng xuất hiện cùng nhau trong tập train, nên số đếm bằng 0 và xác suất MLE bằng 0.

2. Tại sao phải dùng log probability?
- Vì nhân nhiều xác suất nhỏ với nhau sẽ bị tràn số về 0 (underflow). Dùng log biến phép nhân thành phép cộng giúp tính toán an toàn.

3. Perplexity thấp nghĩa là gì?
- Nghĩa là mô hình ít bị bất ngờ, đoán từ tiếp theo tốt hơn và gán xác suất cao hơn cho câu.

4. Tại sao trigram không nhất thiết tốt hơn bigram trên test set?
- Vì trigram bị thưa dữ liệu nặng hơn. Cụm 3 từ mới xuất hiện rất nhiều trên test set nên trigram bị phạt xác suất làm mịn nhiều hơn bigram.

5. Nếu cat eats chưa xuất hiện trong training corpus thì model xử lý thế nào?
- Dùng MLE thì xác suất bằng 0 và câu bằng 0.
- Dùng Laplace smoothing thì gán xác suất dương (1 chia cho C(cat) cộng V) để câu không bị triệt tiêu về 0.
