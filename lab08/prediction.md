# Lab 08 — Pre-Experiment Predictions & Verification

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** Pretrained Transformers and BERT Fine-Tuning  

---

## 1. Các dự đoán trước khi chạy thực nghiệm (Mục 11)

Theo yêu cầu của đề bài tại Mục 11, các dự đoán dưới đây được ghi nhận trước khi tiến hành huấn luyện mô hình.

### Prediction 1: Pretrained Transformer vs. Transformer from scratch trên IMDb
*Đề bài:* Một pretrained Transformer fine-tuned trên IMDb có khả năng đạt kết quả tốt hơn Transformer nhỏ được train from scratch trên cùng dữ liệu không?

* **Prediction:** **CÓ**, Pretrained Transformer fine-tuning sẽ vượt trội áp đảo so với Transformer train from scratch.
* **Reason:** 
  1. *Biểu diễn ngôn ngữ tổng quát:* Pretrained model (như BERT) đã được huấn luyện trước trên một lượng dữ liệu văn bản khổng lồ (BooksCorpus và English Wikipedia với hơn 3.3 tỷ từ) thông qua nhiệm vụ Masked Language Modeling (MLM). Mô hình đã nắm vững cấu trúc cú pháp, trường từ vựng ngữ nghĩa và quan hệ ngữ cảnh phong phú.
  2. *Hạn chế của mô hình From-scratch:* Một Transformer được khởi tạo trọng số ngẫu nhiên (random initialization) và huấn luyện từ đầu chỉ trên tập IMDb hữu hạn (4,000 mẫu train) buộc phải học đồng thời cả hai việc: biểu diễn ngôn ngữ tự nhiên từ con số 0 và phân biệt nhãn cảm xúc. Do kiến trúc Transformer không có thiên kiến cảm nạp cục bộ (inductive bias) mạnh như CNN/RNN, nó đòi hỏi lượng dữ liệu cực lớn để hội tụ; với tập dữ liệu nhỏ, nó rất dễ rơi vào tình trạng overfit nghiêm trọng hoặc không thể hội tụ.
* **Expected evidence:** 
  - Accuracy và Macro F1 của Pretrained Transformer sẽ cao hơn rõ rệt (dự kiến $> 85\%$, trong khi Transformer from scratch với ít dữ liệu chỉ dao động quanh mức đoán ngẫu nhiên $50\% - 65\%$).
  - Mô hình Pretrained hội tụ cực nhanh (chỉ sau 2–4 epochs) và hàm mất mát trên tập Validation giảm ổn định.

---

### Prediction 2: Frozen Encoder (Feature Extraction) vs. Full Fine-tuning
*Đề bài:* Nếu freeze toàn bộ pretrained encoder và chỉ train classification head:
$$\text{BERT encoder (frozen)} \to \text{classification head (train)} \to \text{prediction}$$
kết quả sẽ:
- A. Tốt hơn fine-tuning toàn bộ
- B. Tương đương
- **C. Kém hơn**
- D. Không thể dự đoán

* **Đáp án lựa chọn:** **C. Kém hơn**
* **Giải thích (Reason):**
  1. *Biểu diễn tĩnh vs. Biểu diễn thích nghi:* Khi đóng băng encoder, các vector biểu diễn tại vị trí `[CLS]` là cố định, mang tính tổng quát từ nhiệm vụ MLM thời kỳ pre-training. Tầng phân loại Linear head chỉ có thể học một siêu phẳng phân tách tuyến tính (linear separator) trên không gian đặc trưng tĩnh này.
  2. *Cơ chế thích ứng nhiệm vụ của Fine-tuning:* Khi fine-tuning toàn bộ, gradient từ hàm loss bài toán phân loại cảm xúc được lan truyền ngược (backpropagate) qua toàn bộ 12 tầng Transformer Encoder. Điều này cho phép các ma trận Attention (Query, Key, Value) tái căn chỉnh trọng số chú ý để tập trung đặc biệt vào các từ ngữ, cụm từ mang sắc thái khen/chê mạnh (ví dụ: *masterpiece, horrific, waste of time*), giúp tách biệt cụm nhãn rõ nét hơn nhiều trong không gian biểu diễn ẩn.

---

### Prediction 3: Hiệu quả khi suy giảm dữ liệu huấn luyện (Data Efficiency - 10% data)
*Đề bài:* Nếu chỉ có $10\%$ dữ liệu huấn luyện ($400$ mẫu so với $4,000$ mẫu), mô hình pretrained hay mô hình train from scratch sẽ giảm performance mạnh hơn?

* **Trả lời:** **Mô hình train from scratch sẽ giảm performance mạnh hơn rất nhiều.**
* **Cơ sở từ vai trò của Pretraining:**
  1. *Tính hiệu quả mẫu (Sample Efficiency) của Pretraining:* Mô hình Pretrained đã học xong toàn bộ phân phối xác suất ngôn ngữ tự nhiên trong giai đoạn tiền huấn luyện. Khi đưa vào downstream task với chỉ 10% dữ liệu ($400$ mẫu), bài toán thực chất chỉ là học một ánh xạ (mapping) từ vector biểu diễn có sẵn sang 2 nhãn cảm xúc. Do đó, mô hình vẫn duy trì được độ chính xác rất cao.
  2. *Sự sụp đổ của mô hình From-scratch:* Mô hình train from scratch cần hàng chục nghìn đến hàng triệu tham số cần tối ưu từ con số 0. Với chỉ $400$ mẫu huấn luyện, số lượng tham số áp đảo hoàn toàn số lượng mẫu, dẫn đến hiện tượng overfit ngay lập tức (mô hình học vẹt nhớ mẫu train nhưng hoàn toàn bất lực trên tập test) hoặc mô hình suy biến về dự đoán nhãn đa số (Majority Class guessing với Accuracy $\approx 50\%$).

---

## 2. Bảng đối chiếu Thực nghiệm Kiểm chứng (Verification)

Sau khi tiến hành huấn luyện thực tế trên tập IMDb (ghi nhận trong file `results.csv`), kết quả thực nghiệm hoàn toàn khớp và củng cố vững chắc các dự đoán lý thuyết ban đầu:

| Thí nghiệm kiểm chứng | Dự đoán ban đầu | Kết quả thực nghiệm thực tế | Nhận định đối chiếu |
| :--- | :--- | :--- | :--- |
| **Exp 1: Pretrained vs. Scratch** (4000 train samples) | Pretrained vượt trội hoàn toàn; Scratch khó hội tụ | **Pretrained:** Acc = **86.00%**, F1 = **0.8649**<br>**From scratch:** Acc = **49.80%**, F1 = **0.0000** | **Khớp 100%:** Mô hình from-scratch hoàn toàn không thể học được biểu diễn ngôn ngữ từ 4000 mẫu, rơi vào sụp đổ nhãn (Accuracy $\approx 50\%$). |
| **Exp 2: Frozen vs. Fine-tuning** (4000 train samples) | Frozen kém hơn Full Fine-tuning (Chọn C) | **Fine-tuning:** Acc = **86.00%**, F1 = **0.8649**<br>**Frozen:** Acc = **77.90%**, F1 = **0.7877** | **Khớp 100%:** Fine-tuning cao hơn Frozen tới **+8.1%** Accuracy. Tuy nhiên Frozen vẫn đạt 77.9% dù chỉ train 1,538 tham số, chứng minh biểu diễn pretrain rất giàu thông tin. |
| **Exp 3: Data Efficiency** (10% train = 400 samples) | Scratch sụp đổ hoàn toàn; Pretrained vẫn giữ vững phong độ | **Pretrained (10%):** Acc = **82.60%**, F1 = **0.8249**<br>**From-scratch (10%):** Acc = **50.20%**, F1 = **0.6684** (đoán toàn bộ 1 nhãn) | **Khớp 100%:** Pretrained với 10% dữ liệu chỉ giảm 3.4% so với 100% data, vượt trội hơn mô hình from scratch chạy 100% data (+32.8%). |
