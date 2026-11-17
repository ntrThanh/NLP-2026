# Lab 09 — Predictions Before Experiment

## Prediction 1

**Câu hỏi:** Nếu sử dụng pretrained Transformer thay vì classifier đơn giản dựa trên word features, NER sẽ thay đổi như thế nào?

**Dự đoán:** Pretrained Transformer sẽ cho F1 cao hơn đáng kể so với baseline (Logistic Regression / TF-IDF).

**Lý do:**
- Baseline chỉ nhìn thấy từng token riêng lẻ hoặc các feature cục bộ. Nó không thể suy luận từ ngữ cảnh xung quanh để phân biệt "Paris" là LOC (thành phố) hay PER (tên người Paris Hilton).
- Pretrained Transformer tạo contextual representation: cùng một token "Washington" sẽ có vector khác nhau tùy thuộc vào câu "Washington D.C." (LOC) hay "George Washington" (PER). Khả năng này giúp giải quyết ambiguity mà baseline không thể làm được.
- Ngoài ra, pretrained model đã được huấn luyện trên lượng văn bản lớn nên đã "biết" rằng các từ viết hoa thường là entity, các từ đứng sau "Mr." thường là PER, v.v.

---

## Prediction 2

**Câu hỏi:** Entity dài nhiều token (ví dụ "United Nations Security Council") sẽ dễ hay khó hơn entity 1 token (ví dụ "Microsoft")?

**Dự đoán:** Entity dài sẽ **khó hơn** đáng kể.

**Lý do:**
- Entity 1 token chỉ cần dự đoán đúng 1 tag B-X. Entity 4 token yêu cầu dự đoán đúng cả 4 tag (B-X, I-X, I-X, I-X) — sai bất kỳ tag nào đều khiến entity-level evaluation tính là sai toàn bộ.
- Ranh giới (boundary) của entity dài khó xác định hơn: model có thể nhầm "United Nations" là entity hoàn chỉnh thay vì "United Nations Security Council".
- Entity dài thường hiếm hơn trong training data, model có ít mẫu để học pattern.

---

## Prediction 3

**Câu hỏi:** Entity hiếm xuất hiện trong training data có khả năng gây lỗi nhiều hơn không?

**Dự đoán:** **Có**, entity hiếm sẽ bị lỗi nhiều hơn rõ rệt.

**Lý do:**
- Model học pattern từ dữ liệu huấn luyện. Nếu một entity chỉ xuất hiện 1-2 lần (hoặc không xuất hiện), model không có đủ tín hiệu để nhận diện nó.
- Pretrained Transformer có thể giảm thiểu vấn đề này nhờ contextual representation — dù chưa thấy entity cụ thể, nó vẫn có thể suy luận từ ngữ cảnh (ví dụ: "CEO of Xylophon Corp" → "Xylophon Corp" có khả năng là ORG dù từ "Xylophon" hoàn toàn mới).
- Tuy nhiên, với các entity hiếm mà ngữ cảnh cũng không rõ ràng (ví dụ: tên địa danh lạ), model vẫn sẽ gặp khó khăn đáng kể.
