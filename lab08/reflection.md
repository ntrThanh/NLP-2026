# Báo cáo Phản tư & Tổng kết (Reflection) — LAB 08

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** Pretrained Transformers and BERT Fine-Tuning  
**Tác giả:** Nguyễn Trọng Thành  

---

## Câu hỏi trọng tâm: Tại sao trong thực tế người ta thường không bắt đầu bằng việc train Transformer từ đầu?

Trải qua **LAB 07**, chúng ta đã hiểu rõ cơ chế toán học và tự tay xây dựng từng khối của kiến trúc Transformer Encoder từ các tầng nguyên thủy (Self-Attention, Multi-Head Attention, Residual, LayerNorm, FFN). Tuy nhiên, bước sang **LAB 08**, kết quả thực nghiệm đã đem lại một bài học mang tính nền tảng trong thực tế ứng dụng NLP hiện đại: **Huấn luyện một Transformer từ đầu (train from scratch) cho một tác vụ cụ thể hầu như luôn là giải pháp kém hiệu quả và không khả thi.**

Lý do bắt nguồn từ 6 khía cạnh cốt lõi sau:

1. **Khía cạnh Dữ liệu (Data):** Kiến trúc Transformer hoàn toàn không có các thiên kiến cảm nạp cục bộ (inductive bias) mạnh mẽ như tính bất biến tịnh tiến của CNN hay giả thiết Markov tuần tự của RNN. Do đó, để các ma trận trọng số Query, Key, Value học được cấu trúc cú pháp và phân phối tự nhiên của ngôn ngữ, mô hình đòi hỏi một lượng dữ liệu khổng lồ (hàng tỷ đến hàng chục tỷ từ vựng). Hầu hết các bài toán thực tế trong doanh nghiệp hay nghiên cứu (như phân loại văn bản, trích xuất thông tin) chỉ sở hữu tập dữ liệu gán nhãn rất hạn chế (vài trăm đến vài nghìn mẫu). Như thực nghiệm Exp 1 đã chỉ ra, huấn luyện from-scratch trên 4,000 mẫu IMDb chỉ đạt độ chính xác $49.8\%$ (tương đương đoán ngẫu nhiên).
2. **Biểu diễn Ngôn ngữ Tiền Huấn luyện (Pretrained Representation):** Thông qua mục tiêu tự giám sát Masked Language Modeling (MLM), các mô hình như BERT đã hấp thụ một khối lượng tri thức phổ quát khổng lồ từ toàn bộ Wikipedia và BooksCorpus. Biểu diễn ngữ cảnh (contextual representation) mà BERT tạo ra tại vị trí `[CLS]` đã phân cụm ngữ nghĩa cực kỳ rõ ràng, đóng vai trò như một bộ trích xuất đặc trưng ngôn ngữ toàn diện.
3. **Chi phí Tính toán (Computational Cost):** Việc tiền huấn luyện một mô hình ngôn ngữ lớn đòi hỏi cụm hàng chục, hàng trăm GPU/TPU chạy liên tục trong nhiều tuần với kinh phí hàng trăm nghìn USD và lượng phát thải năng lượng rất lớn. Việc tái sử dụng (reuse) checkpoint đã được pretrain giúp các kỹ sư tiết kiệm đến $99\%$ chi phí điện toán.
4. **Hiệu quả trên Tác vụ Đích (Downstream Task):** Thay vì phải vừa học hiểu ngôn ngữ vừa học giải quyết bài toán, mô hình khởi đầu từ điểm xuất phát tối ưu $\theta_{\text{pretrained}}$ có thể tập trung hoàn toàn vào việc nhận diện mục tiêu của downstream task.
5. **Sự vượt trội của Fine-tuning:** Quá trình fine-tuning chỉ cần từ 2 đến 4 epochs trên một chiếc máy tính cá nhân hoặc GPU phổ thông trong vài phút để đạt độ chính xác $> 86\%$. Tinh chỉnh toàn bộ mô hình cho phép gradient lan truyền ngược tái định hình các đầu chú ý (attention heads) thích ứng nhanh chóng với đặc thù của miền dữ liệu mới.
6. **Nhận thức về Hạn chế & Dịch chuyển Miền (Limitation & Domain Shift):** Mặc dù mô hình pretrained rất mạnh mẽ, nó không phải là "chiếc chìa khóa vạn năng". Mô hình vẫn chịu giới hạn nghiêm trọng về độ dài chuỗi ($O(T^2)$), hiện tượng tự tin quá mức vào các dự đoán sai (overconfidence), nhầm lẫn trước thủ pháp trào phúng (sarcasm), và suy giảm độ chính xác khi đối mặt với sự dịch chuyển phân phối miền dữ liệu (Domain Shift).

---

## Cầu nối chuyển tiếp sang LAB 09 (Bridge to LAB 09)

Trong **LAB 08**, chúng ta sử dụng vector biểu diễn duy nhất tại `[CLS]` để phân loại toàn bộ văn bản (Sequence Classification) — gán **một nhãn duy nhất cho cả câu**.

Tuy nhiên, trong các bài toán thực tiễn như nhận dạng thực thể (Named Entity Recognition - NER) hay trích xuất thông tin y sinh, ta cần xác định từng từ ngữ trong câu ứng với thực thể nào (ví dụ: đâu là Tên người `PERSON`, Địa điểm `LOCATION`, hay Tổ chức `ORG`). Do đó, **LAB 09** sẽ mở rộng việc khai thác Pretrained Transformer từ **Sequence Classification** sang **Token Classification**, tận dụng trọn vẹn toàn bộ các vector ẩn đầu ra $h_1, h_2, \dots, h_T$ của Transformer.
