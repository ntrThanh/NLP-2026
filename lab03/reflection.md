# Báo cáo phản tư và ôn tập (Reflection & Learning Check)
---

## 1. Vấn đề từ đa nghĩa (Polysemy)

Xem xét từ bank trong hai ngữ cảnh cụ thể:
- Câu 1: I deposited money in the bank. (nghĩa tài chính, ngân hàng)
- Câu 2: We sat on the river bank. (nghĩa địa lý, bờ sông)

Mô hình Word2Vec gán cố định đúng một vector duy nhất trong không gian cho mỗi mục từ trong từ vựng. Khi từ bank xuất hiện trong cả hai ngữ cảnh hoàn toàn khác nhau, hàm mất mát sẽ ép vector của từ này phải thỏa hiệp để dự đoán được cả các từ tài chính như money, deposit, cash lẫn các từ địa lý như river, water, stream.

Hệ quả là vector duy nhất của bank sẽ bị kéo về một vị trí trung bình ở giữa hai miền nghĩa, hoặc bị chi phối bởi nghĩa nào xuất hiện nhiều lần hơn trong ngữ liệu. Đây là hạn chế căn bản của các mô hình static word embeddings: không thể biểu diễn chính xác ngữ nghĩa khi từ có nhiều nét nghĩa khác nhau.

---

## 2. Bảng so sánh từ Word2Vec đến Transformer

Bảng so sánh 4 thế hệ biểu diễn ngôn ngữ theo Mục 27:

| Mô hình | Phụ thuộc ngữ cảnh (Context-dependent) | Thưa hay Đặc (Sparse / Dense) | Một từ có nhiều vector? |
| :--- | :--- | :--- | :--- |
| TF-IDF | Không | Sparse | Không |
| Co-occurrence | Không | Sparse | Không |
| Word2Vec | Không | Dense | Không |
| Contextual Embedding (BERT / Transformer) | Có | Dense | Có |

### Tại sao từ bank cần biểu diễn theo ngữ cảnh (Contextual Representation)?

Trong các mô hình Transformer như BERT hay RoBERTa, biểu diễn của một từ được tính toán động thông qua cơ chế Self-Attention. Khi từ bank đi cùng từ river, cơ chế chú ý sẽ tập trung trọng số vào các từ xung quanh để tạo ra một vector mang đậm đặc trưng dòng sông. Ngược lại, khi bank đi cùng money, vector tạo ra sẽ mang đặc trưng tài chính. Cơ chế này giúp giải quyết triệt để bài toán từ đa nghĩa mà Word2Vec tĩnh không thể làm được.

---

## 3. Tuyên bố sử dụng AI (AI Assistance Statement)

Tuân thủ Mục 28 của đề bài W3.pdf:
- Công cụ sử dụng: Google Antigravity IDE (Gemini 3.8 Flash).
- Mục đích sử dụng: Hỗ trợ kiểm tra lỗi cú pháp, gợi ý cấu trúc code thực nghiệm Word2Vec, tối ưu hóa thời gian chạy và hỗ trợ vẽ biểu đồ trực quan hóa dữ liệu.
- Nội dung do AI tạo: Các hàm hỗ trợ đo đạc thời gian, code giảm chiều PCA bằng NumPy và khung trình bày bảng Markdown.
- Nội dung sinh viên tự làm và hiệu chỉnh: Toàn bộ bài làm tính toán trên giấy, các nhận định và suy luận khoa học trong phần nhận xét, đối chiếu ngữ liệu thực tế và phân tích lỗi.
- Quy trình kiểm chứng: Chạy kiểm thử độc lập toàn bộ mã nguồn trên môi trường thực tế, kiểm tra tính toán tay và xác nhận mọi kết quả đều khớp với dữ liệu thật.

---

## 4. Chuẩn bị câu hỏi vấn đáp cá nhân (Individual Learning Check)

Trả lời ngắn gọn 6 câu hỏi trọng tâm theo Mục 29:

### Câu 1: Distributional hypothesis là gì?
Giả thuyết phân phối khẳng định rằng những từ xuất hiện trong các ngữ cảnh tương tự nhau thường có ý nghĩa tương tự nhau. Đây là nền tảng cốt lõi để xây dựng các mô hình word embeddings từ dữ liệu văn bản thô mà không cần gán nhãn thủ công.

### Câu 2: Tại sao doctor và physician có thể gần nhau?
Hai từ này gần nhau vì chúng xuất hiện trong cùng các mẫu ngữ cảnh cú pháp và kết hợp ngữ nghĩa tương tự nhau, chẳng hạn như cùng đứng sau các động từ khám bệnh và cùng làm chủ ngữ cho hành động kê đơn, điều trị.

### Câu 3: CBOW khác Skip-gram ở đâu?
CBOW sử dụng túi các từ ngữ cảnh xung quanh để dự đoán từ đích ở giữa, trong khi Skip-gram làm ngược lại: lấy từ đích ở giữa để dự đoán từng từ ngữ cảnh xung quanh.

### Câu 4: Tại sao tăng context window có thể vừa tốt vừa xấu?
Cửa sổ lớn giúp mô hình học được các liên kết ngữ nghĩa theo chủ đề rộng, nhưng mặt xấu là làm mất đi tính chính xác về mặt cú pháp và khiến vector dễ bị nhiễu bởi các hư từ xuất hiện thường xuyên.

### Câu 5: Tại sao Word2Vec không phân biệt được hai nghĩa của bank?
Vì Word2Vec là mô hình tĩnh gán một vector duy nhất cho mỗi từ trong từ điển, khiến vector của bank bị ép thành điểm trung bình giữa nghĩa ngân hàng và bờ sông.

### Câu 6: Tại sao TF-IDF không phải word embedding?
TF-IDF là biểu diễn đếm thưa cho cả văn bản dựa trên tần số xuất hiện, trong đó các chiều vector đại diện cho từ vựng và hai từ đồng nghĩa vẫn có hai chiều vector vuông góc hoàn toàn với nhau, không phản ánh được độ tương đồng ngữ nghĩa.
