# Lab 10 — Predictions Before Experiment

> Các dự đoán dưới đây được ghi **trước khi** chạy bất kỳ thực nghiệm nào. Phần "Đối chiếu sau thực nghiệm" ở cuối file được bổ sung sau khi có kết quả và không sửa lại nội dung dự đoán ban đầu.

---

## Prediction 1 — Câu hỏi và context có nhiều từ trùng nhau

**Câu hỏi:** Nếu question và context chứa nhiều từ giống nhau nhưng chỉ một vị trí thực sự trả lời câu hỏi, model có dễ bị nhầm không?

**Dự đoán:**
- **Baseline keyword-overlap sẽ bị nhầm nặng.** Nó chỉ đếm số từ chung, nên câu nào lặp lại nhiều từ của câu hỏi nhất sẽ được chọn, kể cả khi câu đó không chứa đáp án.
- **Fine-tuned Transformer bị ảnh hưởng ít hơn nhưng không miễn nhiễm.** Self-attention giữa question và context cho phép model khớp *quan hệ* (ai làm gì, khi nào) chứ không chỉ khớp từ. Tuy nhiên, SQuAD được tạo bởi annotator nhìn vào đoạn văn nên câu hỏi thường "chép" từ ngữ của câu chứa đáp án → model có xu hướng học shortcut "tìm câu giống câu hỏi nhất rồi lấy cụm có đúng kiểu thực thể". Tôi dự đoán:
  - Câu hỏi có **lexical overlap cao** với câu chứa đáp án: F1 cao hơn rõ rệt (chênh khoảng 8–15 điểm F1) so với nhóm overlap thấp.
  - Lỗi phổ biến là chọn đúng *kiểu* đáp án (một năm, một tên người) nhưng ở **câu khác** có nhiều từ trùng (distractor).

---

## Prediction 2 — Answer span dài

**Câu hỏi:** Answer span càng dài thì boundary prediction có khó hơn không?

**Dự đoán:** **Có — EM giảm mạnh theo độ dài, F1 giảm chậm hơn.**
- Với answer 1 token, start và end trùng nhau; model chỉ cần chọn đúng 1 vị trí. Với answer 7+ token (một mệnh đề, một cụm mô tả) có nhiều điểm dừng "hợp lý" (có lấy cụm giới từ phía sau không? có lấy mệnh đề phụ không?) → rất dễ lệch boundary.
- EM nhóm 1 token dự đoán ~85–90%, nhóm 7+ token dự đoán chỉ ~40–55%.
- F1 sẽ giảm ít hơn EM vì prediction lệch boundary vẫn chồng lấn nhiều với gold (partial credit) → khoảng cách F1 − EM sẽ **tăng** theo độ dài answer.
- Lỗi wrong-end sẽ nhiều hơn wrong-start, vì điểm kết thúc của các cụm dài (mệnh đề, cụm danh từ có bổ ngữ) mơ hồ hơn điểm bắt đầu (thường ngay sau giới từ hoặc động từ).

---

## Prediction 3 — Context chứa nhiều thông tin không liên quan / distractor

**Câu hỏi:** Nếu context chứa nhiều thông tin không liên quan, performance thay đổi như thế nào?

**Dự đoán:**
- **Thêm đoạn văn không liên quan (khác chủ đề):** performance giảm **nhẹ** (≈ 1–3 điểm F1). Model đã học để nhận biết vùng liên quan qua attention với câu hỏi; đoạn văn khác chủ đề ít có từ trùng nên ít gây nhiễu. Ảnh hưởng chính đến từ việc context dài hơn → nhiều window hơn → nhiều ứng viên hơn khi tổng hợp kết quả trên các window.
- **Thêm câu distractor có nhiều từ trùng với câu hỏi** (giống adversarial SQuAD — AddSent): performance giảm **mạnh** (có thể 10–20+ điểm F1), vì đây chính là shortcut lexical overlap mà model dựa vào.
- Baseline keyword-overlap sẽ sụp đổ trước distractor kiểu này vì nó chỉ dựa trên overlap.

---

## Prediction 4 — Thay đổi thứ tự câu trong context

**Câu hỏi:** Nếu chỉ thay đổi thứ tự một số câu trong context nhưng giữ nguyên answer, model có bị ảnh hưởng không?

**Dự đoán:** **Có, nhưng nhỏ** (≈ 1–4 điểm F1).
- Phần lớn câu hỏi SQuAD trả lời được bằng **một câu duy nhất**; nếu câu chứa đáp án còn nguyên vẹn thì việc dời vị trí của nó không thay đổi thông tin cục bộ mà model cần.
- Tuy nhiên model sẽ bị ảnh hưởng ở:
  - Câu hỏi cần **đồng tham chiếu** giữa các câu ("He", "It", "The school" ở câu sau trỏ về thực thể ở câu trước) — xáo trộn làm đứt chuỗi tham chiếu.
  - Câu hỏi cần **suy luận đa câu**.
  - Position embeddings: model có thể đã học thiên lệch vị trí nhẹ (đáp án thường ở nửa đầu đoạn văn).
- Kết luận dự đoán: model "gần như" bất biến với thứ tự câu, cho thấy nó dựa chủ yếu vào khớp cục bộ question–sentence hơn là hiểu cấu trúc toàn đoạn.

---

## Prediction bổ sung — Context length (Experiment 1)

**Dự đoán:** Tăng `max_length` từ 128 → 256 → 384 **không** cải thiện tuyến tính.
- 128 → 256: cải thiện rõ (≈ +3–6 F1), vì với 128 token, sau khi trừ câu hỏi (~15 token) và special tokens, mỗi window chỉ còn ~110 token context; nhiều đáp án bị cắt ngang ranh giới window hoặc mất ngữ cảnh giải thích nằm ở câu khác.
- 256 → 384: cải thiện nhỏ hoặc không đáng kể (đoạn văn SQuAD trung bình ~140 từ ≈ 160–180 token, phần lớn đã nằm vừa trong một window 256).
- Chi phí: 384 tốn thời gian huấn luyện nhiều nhất trên mỗi feature, nhưng 128 tạo **nhiều feature hơn** (nhiều window/ví dụ), nên chênh lệch tổng thời gian nhỏ hơn mức tỷ lệ với độ dài.

---

## Đối chiếu sau thực nghiệm

*(Bổ sung sau khi chạy — xem `README.md` mục 3 và `error_analysis.md`.)*
