# Lab 10 — Reflection

**Câu hỏi:** LAB 08, LAB 09 và LAB 10 đều sử dụng pretrained Transformer nhưng giải quyết ba bài toán khác nhau như thế nào?

```
                        Pretrained Transformer (bert-base-cased)
                     H = Encoder(input) ∈ R^{T × 768}
                                 │
                ┌────────────────┼─────────────────┐
                ↓                ↓                 ↓
       [CLS] representation   Token reps        Token reps
            h_[CLS] ∈ R^768   h_1 … h_T         h_1 … h_T  (question + context)
                ↓                ↓                 ↓
        Linear(768 → C)    Linear(768 → 9)    Linear(768 → 2)
                ↓            mỗi token           ↓          ↓
          softmax trên C     softmax trên     softmax     softmax
             lớp              9 nhãn BIO      trên T vị trí (start / end)
                ↓                ↓                 ↓
         Classification         NER            Start / End → span
                ↓                ↓                 ↓
             LAB 08           LAB 09            LAB 10
```

Ba lab gần như dùng **cùng một backbone** và cùng một quy trình fine-tune (AdamW, warmup, 2–3 epoch, learning rate cỡ 3e-5). Thứ thay đổi chỉ là **ta đọc biểu diễn nào** và **target có hình dạng gì**.

Ở **LAB 08**, cả văn bản được nén vào vector `[CLS]` và một lớp linear chọn một trong C lớp. Target là một nhãn duy nhất cho mỗi văn bản. Thông tin vị trí bị bỏ đi hoàn toàn, nên model chỉ trả lời được câu hỏi "văn bản này thuộc loại gì".

Ở **LAB 09**, ta giữ lại vector của **mọi** token. Cùng một lớp linear được áp dụng độc lập lên từng vị trí, nên softmax chạy theo **chiều nhãn**: mỗi token chọn một trong 9 nhãn BIO. Khó khăn mới là căn chỉnh nhãn mức từ với subword, giải quyết bằng `word_ids()` và `-100`.

Ở **LAB 10**, lớp linear chỉ có **2 đầu ra** cho mỗi token, nhưng softmax chạy theo **chiều vị trí**. Cả chuỗi phải "bỏ phiếu" chọn ra một token làm start và một token làm end. "Số lớp" của bài toán chính là độ dài chuỗi. Input cũng thay đổi: question và context được ghép trong một chuỗi để self-attention cho mỗi token context "nhìn thấy" câu hỏi. Đây là điều baseline keyword-overlap không làm được (F1 28.6 so với 88.7). Khó khăn kỹ thuật chuyển từ căn chỉnh nhãn sang **căn chỉnh vị trí**: dataset cho vị trí ký tự, model cần vị trí token, và chỉ `offset_mapping` mới nối hai hệ chỉ số này. Thêm vào đó, context dài phải được chia thành nhiều window chồng lấn (`stride`), rồi gom kết quả về lại câu hỏi qua `overflow_to_sample_mapping`.

Vì sao cùng backbone lại làm được ba việc? Pretraining (masked language modeling) buộc mỗi `h_t` mã hóa ngữ cảnh hai chiều quanh token t, tức là "token này là gì, đóng vai trò gì trong câu, liên quan thế nào với các token khác". Thông tin đó đủ để suy ra cả chủ đề văn bản, loại thực thể của từng từ, lẫn việc "từ này có phải nơi câu trả lời bắt đầu hay không". Prediction head chỉ là một phép chiếu tuyến tính rất nhỏ (với LAB 10 là 768 × 2 + 2 = 1,538 tham số so với 108 triệu tham số của encoder). Nó **định nghĩa câu hỏi** ta đặt cho biểu diễn, còn fine-tuning điều chỉnh nhẹ backbone để biểu diễn phục vụ câu hỏi đó tốt hơn. Kết quả học được nhanh cũng cho thấy điều này: chỉ sau 1/4 epoch model QA đã đạt F1 75.6 trên dev.

Thực nghiệm cũng cho thấy giới hạn của cách tiếp cận span: 58% lỗi là lỗi boundary, EM giảm từ 85% xuống 62% với đáp án dài 7+ từ, và model **luôn** trả về một span, kể cả khi context không chứa đáp án (câu hỏi về World Cup trong `document_qa.py`). Hơn nữa, LAB 10 giả định context đã được cho sẵn. Trong thực tế ta phải **tìm** context trước (semantic retrieval, LAB 11) rồi mới trích xuất hoặc sinh câu trả lời (RAG, LAB 12). Chuỗi classification → token prediction → span prediction → retrieval → RAG vì thế là chuỗi các cách đặt **head và target** khác nhau lên cùng một ý tưởng: biểu diễn ngữ cảnh do Transformer tạo ra.
