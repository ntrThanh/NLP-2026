# Lab 09 — Reflection & Bridge to Lab 10

## 1. Unified Contextual Representations across Downstream Tasks

Một trong những đóng góp mang tính cách mạng của kiến trúc Pretrained Transformer (như BERT) là tính thống nhất của không gian biểu diễn ngữ cảnh (unified contextual representations). Cùng một mô hình nền tảng đã học qua giai đoạn pre-training (Masked Language Modeling) có thể giải quyết cả bài toán phân loại toàn bộ chuỗi văn bản (Sequence Classification - Lab 08) lẫn bài toán gán nhãn từng phần tử trong chuỗi (Token Classification / NER - Lab 09) chỉ bằng việc thay thế prediction head ở tầng cuối cùng.

Cơ chế này được trực quan hóa qua luồng xử lý:

```
                  Pretrained Transformer Backbone (BERT)
                                   ↓
            Contextual Token Representations [h_0, h_1, ..., h_n]
                                   ↓
                 ┌─────────────────┴─────────────────┐
                 ↓                                   ↓
        [CLS] Representation (h_0)       Every Token Repr. (h_1, ..., h_n)
                 ↓                                   ↓
     Linear Classification Head            Linear Token Classifier Head
                 ↓                                   ↓
      Sequence Label (y ∈ C)                Token Labels (y_i ∈ BIO-Tags)
    (Lab 08: Sentiment, Topics)               (Lab 09: NER Entities)
```

Bản chất của Transformer là cơ chế Multi-Head Self-Attention. Sau $L$ lớp Transformer Blocks, mỗi vị trí $i$ trong câu nhận được một vector ẩn $\mathbf{h}_i \in \mathbb{R}^d$ ($d=768$). Vector này không chỉ mang đặc trưng ngữ nghĩa của riêng từ tại vị trí $i$ mà đã hấp thụ trọn vẹn thông tin ngữ cảnh toàn cục từ tất cả các vị trí khác trong chuỗi input:
1. **Trong Sequence Classification (Lab 08):** Token đặc biệt `[CLS]` được thiết kế để tương tác hai chiều với mọi token khác thông qua self-attention qua tất cả các lớp. Do đó, vector $\mathbf{h}_{\text{CLS}} = \mathbf{h}_0$ hoạt động như một biểu diễn tóm lược cô đọng toàn bộ ý nghĩa ngữ nghĩa của cả câu. Ta chỉ cần áp dụng một ma trận chiếu tuyến tính $\mathbf{W}_{\text{seq}} \in \mathbb{R}^{K \times d}$ lên $\mathbf{h}_0$ để dự đoán phân phối xác suất lớp cho toàn câu.
2. **Trong Token Classification (Lab 09):** Bài toán NER đòi hỏi xác định nhãn ngữ pháp/thực thể cho từng vị trí. Thay vì chỉ lấy `[CLS]`, ta lấy toàn bộ chuỗi vector ẩn $[\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_n]$. Mỗi vector $\mathbf{h}_i$ phản ánh chính xác từ đó đang đóng vai trò gì trong cấu trúc ngữ cảnh của câu. Bằng cách áp dụng cùng một ma trận chiếu $\mathbf{W}_{\text{token}} \in \mathbb{R}^{T \times d}$ ($T = 9$ nhãn BIO) lên từng vector $\mathbf{h}_i$, ta thu được phân phối nhãn BIO cho từng token.

Sự linh hoạt này chứng minh rằng năng lực cốt lõi của Transformer nằm ở tầng mã hóa ngữ cảnh (encoder backbone), còn bài toán cụ thể ở hạ tầng chỉ phụ thuộc vào cách ta trích xuất và định tuyến các vector biểu diễn này ở đầu ra.

---

## 2. Bridge to LAB 10: From Token Classification to Span Extraction

Trong Lab 09, mô hình trích xuất thực thể bằng cách phân lớp rời rạc từng token với nhãn BIO:
$$\text{Text} \longrightarrow \text{Entity Tokens (BIO tags)}$$

Tuy nhiên, trong các bài toán NLP phức tạp hơn như **Hỏi - Đáp dựa trên ngữ cảnh (Extractive Question Answering)**, ta đối mặt với một định dạng bài toán mới:
- **Ngữ cảnh (Context):** *"Paris is the capital of France."*
- **Câu hỏi (Question):** *"What is the capital of France?"*
- **Câu trả lời (Answer):** *"Paris"*

Ở đây, mục tiêu không phải là phân loại thể loại của câu (Sequence Classification) và cũng không phải gán nhãn toàn bộ các từ theo danh mục cố định có sẵn (Token Classification). Thay vào đó, mô hình phải tiếp nhận cặp chuỗi $[\text{Question}; \text{Context}]$ và xác định trực tiếp **vị trí bắt đầu (start token)** và **vị trí kết thúc (end token)** của đoạn văn bản (span) chứa câu trả lời nằm trong Context:

$$\text{Question} + \text{Context} \longrightarrow \text{Pretrained Transformer} \longrightarrow \text{Answer Span } [s, e]$$

Thay vì phân loại $T$ nhãn BIO, prediction head ở Lab 10 sẽ là hai bộ phân loại tuyến tính dự đoán xác suất $P(\text{start} = i)$ và $P(\text{end} = j)$. Sự dịch chuyển từ Token Classification sang Span Prediction sẽ là trọng tâm thực nghiệm của Lab 10.
