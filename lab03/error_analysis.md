# Phân tích lỗi mô hình Word2Vec (Error Analysis)
---

## 1. Ba trường hợp dự đoán đúng (Correct Similarities)

### Trường hợp 1: doctor và physician
- Kết quả quan sát: Độ tương đồng Cosine đạt 0.7293 (xếp hạng cao nhất trong các từ liên quan đến doctor).
- Kỳ vọng: Hai từ là từ đồng nghĩa hoàn toàn, chỉ nghề nghiệp bác sĩ y khoa, kỳ vọng độ tương đồng rất cao.
- Giải thích: Hai từ này có quan hệ thay thế đồng vị. Chúng xuất hiện trong cùng các vị trí cú pháp như đi sau các động từ khám bệnh (see a doctor / consult a physician) và cùng làm chủ ngữ cho các hành động kê đơn thuốc, chẩn đoán bệnh. Mô hình CBOW dự đoán từ đích dựa trên túi từ ngữ cảnh, nên khi gặp các từ ngữ cảnh quen thuộc như care, appointment, hospital thì cả hai từ đều nhận xác suất cao.
- Bằng chứng ngữ liệu: Cả hai từ cùng xuất hiện luân phiên trong các tài liệu y tế với các từ ngữ cảnh chung như patient, clinic, treatment.

### Trường hợp 2: king và queen
- Kết quả quan sát: Độ tương đồng Cosine đạt 0.7673.
- Kỳ vọng: Hai từ chỉ người trị vì hoàng gia, chỉ khác biệt về giới tính, kỳ vọng độ tương đồng rất cao.
- Giải thích: Hai từ cùng chia sẻ khung ngữ nghĩa hoàng gia và quyền lực chính trị. Chúng cùng xuất hiện bên cạnh các từ như palace, royal, crown, reign, kingdom.
- Bằng chứng ngữ liệu: Các đoạn văn bản lịch sử và văn hóa trong ngữ liệu sử dụng king và queen trong các cấu trúc mô tả ngai vàng và hoàng gia tương tự nhau.

### Trường hợp 3: hospital và clinic
- Kết quả quan sát: Độ tương đồng Cosine đạt 0.7036 (từ gần nhất với hospital).
- Kỳ vọng: Hai cơ sở y tế khám chữa bệnh, kỳ vọng độ tương đồng cao.
- Giải thích: hospital và clinic có chức năng tương đương trong việc tiếp nhận bệnh nhân và điều trị. Chúng xuất hiện trong cùng các cụm từ chỉ địa điểm y tế.
- Bằng chứng ngữ liệu: Ngữ liệu chứa nhiều câu mô tả việc chuyển viện hoặc thăm khám tại các cơ sở y tế, trong đó hospital và clinic thay thế lẫn nhau.

---

## 2. Ba trường hợp sai hoặc bất ngờ (Unexpected / Failure Cases)

### Trường hợp 1: car và automobile
- Kết quả quan sát: Độ tương đồng Cosine chỉ đạt 0.2919 (thấp bất ngờ đối với hai từ đồng nghĩa).
- Kỳ vọng: car và automobile là từ đồng nghĩa trực tiếp, kỳ vọng độ tương đồng trên 0.70.
- Giải thích: Nguyên nhân chính là do sự chênh lệch tần số xuất hiện quá lớn trong ngữ liệu. Từ car là từ thông dụng xuất hiện gần 500 lần trong mẫu dữ liệu, trong khi automobile chỉ xuất hiện khoảng 25 lần và chủ yếu nằm trong tên tổ chức hoặc ngữ cảnh bảo hiểm xe hơi cũ. Dữ liệu huấn luyện nhỏ khiến vector của automobile chưa hội tụ đầy đủ.
- Bằng chứng ngữ liệu: car xuất hiện trong các câu sinh hoạt hàng ngày (drive a car, car accident, buy a car), trong khi automobile chỉ xuất hiện trong các cụm như automobile association, automobile insurance.

### Trường hợp 2: doctor và child
- Kết quả quan sát: Độ tương đồng Cosine đạt 0.6887 (cao hơn cả surgeon và therapist khi xét với doctor).
- Kỳ vọng: doctor và child thuộc hai miền khái niệm khác nhau (nghề nghiệp và độ tuổi), kỳ vọng độ tương đồng ở mức trung bình thấp.
- Giải thích: Hiện tượng này bắt nguồn từ thiên kiến chủ đề trong ngữ liệu. Tập dữ liệu chứa nhiều bài viết y tế cộng đồng nói về sức khỏe trẻ em, tiêm chủng và bác sĩ nhi khoa. Sự đồng xuất hiện thường xuyên giữa bác sĩ và trẻ em trong cùng câu văn đã kéo hai vector lại gần nhau.
- Bằng chứng ngữ liệu: Xuất hiện nhiều câu như parents should take their child to see a doctor for routine checkups hoặc the doctor examined the sick child.

### Trường hợp 3: banana và oregano
- Kết quả quan sát: Độ tương đồng Cosine đạt 0.8106 (từ gần nhất với banana).
- Kỳ vọng: banana là trái cây ngọt, oregano là loại rau gia vị mặn, kỳ vọng hai từ không quá gần nhau.
- Giải thích: Từ banana chỉ xuất hiện 14 lần trong toàn bộ tập huấn luyện. Kích thước cửa sổ bằng 5 gom chung các từ ngữ cảnh liên quan đến nấu ăn, công thức chế biến và nguyên liệu nhà bếp. Do số lượng mẫu quá ít, mô hình chỉ học được rằng cả hai từ đều là nguyên liệu thực phẩm mà không phân biệt được hương vị hay phân loại ẩm thực.
- Bằng chứng ngữ liệu: Cả hai từ xuất hiện trong các bài viết về công thức nấu ăn, mẹo làm bếp và danh sách nguyên liệu thực phẩm.

---

## 3. Tổng kết các nguyên nhân gây sai lệch

Từ thực nghiệm trên, các nguyên nhân chính dẫn đến sai lệch biểu diễn vector gồm:
1. Tần suất xuất hiện thấp khiến vector chưa học đủ đặc trưng phân phối.
2. Kích thước tập ngữ liệu giới hạn không bao quát hết các ngữ cảnh đa dạng của từ.
3. Thiên kiến chủ đề của ngữ liệu kéo các từ thuộc cùng bối cảnh lại gần nhau quá mức.
4. Cửa sổ ngữ cảnh rộng gom cả các liên kết chủ đề lỏng lẻo thay vì giữ ranh giới ngữ pháp chặt chẽ.
