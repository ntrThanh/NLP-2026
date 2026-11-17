# Lab 09 — Named Entity Recognition & Information Extraction

Bài thực hành môn Xử lý Ngôn ngữ Tự nhiên (NLP) — Khoa CNTT, ĐHKHTN - ĐHQGHN.

---

## 1. Danh mục sản phẩm (Deliverables Checklist)

Đầy đủ tất cả các thành phần theo đúng yêu cầu tại **Mục 29** của đề cương môn học:

```
lab09/
├── README.md                     # Tài liệu tổng quan, kết quả thực nghiệm và hướng dẫn
├── calculations.md               # Lời giải chi tiết bài tập tính toán 1, 2 và 3
├── prediction.md                 # Các dự đoán giả thuyết trước khi chạy thực nghiệm
├── baseline_ner.py               # Mô hình Baseline (Logistic Regression + Orthographic Features)
├── tokenizer_alignment.ipynb     # Notebook trực quan hóa căn chỉnh nhãn subword (đã chạy)
├── ner_finetuning.ipynb          # Notebook huấn luyện, đánh giá BERT và ứng dụng IE (đã chạy)
├── results.csv                   # Bảng tổng hợp so sánh Baseline vs Transformer
├── entity_metrics.csv            # Bảng chi tiết Precision, Recall, F1 theo từng loại thực thể
├── entity_length_analysis.png    # Đồ thị F1 score theo độ dài thực thể (1, 2, 3, 4+ tokens)
├── confusion_matrix.png          # Ma trận nhầm lẫn chuẩn hóa token-level của mô hình BERT
├── error_analysis.md             # Phân tích 15 mẫu lỗi tiêu biểu và phân tích đa nghĩa ngữ cảnh
├── information_extraction.py     # Pipeline trích xuất thực thể có cấu trúc từ văn bản thô
└── reflection.md                 # Bài thu hoạch 300-500 từ về biểu diễn ngữ cảnh và cầu nối Lab 10
```

---

## 2. Thiết lập thực nghiệm & Môi trường tái lập

Thực nghiệm được huấn luyện và kiểm thử trên máy chủ GPU từ xa:
- **Server:** `server22` (NVIDIA GeForce RTX 3060 12GB VRAM).
- **Môi trường:** Conda environment `works` (`Python 3.10`, `PyTorch 2.5`, `Transformers`, `Datasets`, `Seqeval`, `Scikit-Learn`).
- **Tập dữ liệu:** CoNLL-2003 (English NER benchmark) nạp qua HuggingFace Parquet format (`lhoestq/conll2003`).

### Siêu tham số huấn luyện (Reproducibility Hyperparameters)
- **Kiến trúc:** `bert-base-cased` (12 layers, 768 hidden, 12 heads, 108M tham số) + `AutoModelForTokenClassification` (Linear head 768 -> 9 classes).
- **Độ dài chuỗi tối đa (`max_seq_length`):** 128 tokens.
- **Batch size:** 16 (train), 32 (eval/test).
- **Optimizer:** AdamW, learning rate = $3 \times 10^{-5}$, weight decay = $0.01$.
- **Learning rate scheduler:** Linear warmup over 10% steps.
- **Số epoch:** 3 epochs.
- **Căn chỉnh nhãn:** Strategy A — Gán nhãn cho subword đầu tiên của từ, các subword tiếp theo và token đặc biệt được gán `-100` (`IGNORE_INDEX`).
- **Random seed:** 42.

---

## 3. Kết quả thực nghiệm

### 3.1. So sánh Baseline vs Pretrained Transformer (Test Set)

| Model | Architecture | Precision | Recall | Entity F1 |
|---|---|:---:|:---:|:---:|
| **Baseline** | Logistic Regression (SAGA) + Orthographic Features | 0.6940 | 0.7408 | 0.7166 |
| **Transformer** | `bert-base-cased` + Linear Token Classifier | **0.9054** | **0.9196** | **0.9124** |

> **Nhận xét:** BERT vượt trội mô hình Baseline +19.58% F1. Cơ chế Bidirectional Self-Attention giải quyết triệt để vấn đề phụ thuộc ngữ cảnh dài mà cửa sổ trượt n-gram hạn chế của mô hình Baseline không thể nắm bắt.

### 3.2. Đánh giá chi tiết theo loại thực thể (Entity-Level Breakdown)

| Loại thực thể | Support | Precision | Recall | Entity F1 |
|---|:---:|:---:|:---:|:---:|
| **PER** (Người) | 1,615 | 0.9688 | 0.9604 | **0.9646** |
| **LOC** (Địa điểm) | 1,666 | 0.9230 | 0.9352 | **0.9290** |
| **ORG** (Tổ chức) | 1,661 | 0.8919 | 0.9043 | **0.8981** |
| **MISC** (Khác / Hỗn hợp) | 702 | 0.7628 | 0.8248 | **0.7926** |

> **Nhận xét:** Đúng như Dự đoán 3 (`prediction.md`), thực thể `PER` và `LOC` có độ chính xác cao nhất (>92-96%) do tính nhất quán về ngữ nghĩa và cú pháp. Ngược lại, lớp `MISC` đạt F1 thấp nhất (0.7926) vì tập dữ liệu ít mẫu hơn và ranh giới định nghĩa mờ nhạt (tên sự kiện, ngôn ngữ, quốc tịch, giải đấu).

### 3.3. Đánh giá theo độ dài thực thể (Entity Length Analysis)

| Độ dài thực thể | Số lượng | Entity F1 |
|---|:---:|:---:|
| **1 token** | 3,923 | 0.9115 |
| **2 tokens** | 1,326 | **0.9220** |
| **3 tokens** | 247 | 0.8745 |
| **4+ tokens** | 148 | **0.8444** |

> **Nhận xét:** Đúng như Dự đoán 2 (`prediction.md`), hiệu năng nhận diện thực thể giảm dần khi độ dài thực thể tăng lên (F1 giảm từ 0.9220 xuống 0.8444 đối với thực thể 4+ tokens). Việc duy trì nhất quán chuỗi nhãn `B-X, I-X, I-X, ...` trên nhiều token liên tiếp gặp xác suất suy giảm tích lũy khi không có tầng ràng buộc chuỗi (CRF).

---

## 4. Kiểm tra kiến thức cá nhân (Individual Learning Check)

1. **Sequence classification khác token classification như thế nào?**  
   Sequence classification gán một nhãn duy nhất cho toàn bộ câu/đoạn văn bản (thông qua vector biểu diễn của `[CLS]`). Token classification gán một nhãn riêng biệt cho từng token trong chuỗi (thông qua vector ẩn $\mathbf{h}_i$ của từng token).
2. **BIO tagging dùng để làm gì?**  
   Dùng để biểu diễn ranh giới thực thể trong chuỗi token rời rạc: `B-` (Begin) đánh dấu bắt đầu thực thể, `I-` (Inside) đánh dấu các token bên trong thực thể, và `O` (Outside) đánh dấu các token không thuộc thực thể nào.
3. **B-PER khác I-PER như thế nào?**  
   `B-PER` chỉ định token đầu tiên bắt đầu một thực thể người mới. `I-PER` là các token nối tiếp trong cùng một thực thể người đó. Sự phân biệt này giúp tách biệt chính xác hai thực thể người đứng liền kề nhau (ví dụ: *"John Mary"* gồm `B-PER` và `B-PER`).
4. **Tại sao tokenizer gây ra label alignment problem?**  
   Tokenizer subword (WordPiece, BPE) thường bẻ một từ nguyên vẹn thành nhiều mảnh subword (ví dụ: `HuggingFace` thành `Hug`, `##ging`, `##Face`), trong khi nhãn ban đầu của bộ dữ liệu được gán trên cấp độ từ (word-level). Ta phải căn chỉnh để xác định mảnh subword nào mang nhãn thực thể và mảnh nào bị bỏ qua.
5. **IGNORE trong loss có ý nghĩa gì?**  
   Quy ước gán nhãn `-100` (`IGNORE_INDEX`) cho các token phụ hoặc token đặc biệt (`[CLS]`, `[SEP]`, `[PAD]`) để `CrossEntropyLoss` tự động bỏ qua chúng, không tính vào hàm mất mát và không lan truyền ngược gradient làm lệch trọng số mô hình.
6. **Tại sao NER không nên chỉ đánh giá bằng token accuracy?**  
   Do hiện tượng mất cân bằng nhãn cực độ: hơn 85% các token trong câu là nhãn `O`. Một mô hình tầm thường chỉ dự đoán toàn bộ nhãn `O` vẫn đạt accuracy > 85%, nhưng hoàn toàn vô dụng trong việc trích xuất thực thể.
7. **Entity-level F1 khác token-level F1 như thế nào?**  
   Token-level F1 tính điểm đúng/sai độc lập trên từng token đơn lẻ. Entity-level F1 (chuẩn CoNLL qua seqeval) đòi hỏi mô hình phải dự đoán chính xác cả loại thực thể lẫn toàn bộ ranh giới (start và end) của thực thể đó thì mới được tính là True Positive.
8. **Vì sao entity boundary khó?**  
   Thực thể có độ dài biến thiên, thường đi kèm các từ định tố, bổ ngữ, giới từ hoặc liên từ khiến ranh giới bắt đầu/kết thúc dễ bị cắt cụt hoặc mở rộng quá đà.
9. **Vì sao contextual representation hữu ích cho NER?**  
   Cơ chế Self-Attention cho phép biểu diễn của một từ phụ thuộc vào các từ bao quanh nó trong câu, giúp phân biệt được hiện tượng hoán dụ và đa nghĩa (ví dụ: "Barcelona" là địa điểm tổ chức hội nghị hay câu lạc bộ bóng đá đang thi đấu).
10. **NER có thể được sử dụng như thế nào trong Information Extraction?**  
    NER là bước khởi đầu quan trọng giúp trích xuất các thực thể định danh cốt lõi (Ai, Ở đâu, Khi nào, Tổ chức nào). Dữ liệu thực thể này là đầu vào trực tiếp cho các tác vụ tiếp theo như trích xuất quan hệ (Relation Extraction), trích xuất sự kiện (Event Extraction), và xây dựng Đồ thị tri thức (Knowledge Graph).

---

## 5. Hướng dẫn chạy mã nguồn (How to Run)

### Chạy Baseline NER
```bash
python baseline_ner.py
```

### Chạy huấn luyện và đánh giá toàn diện mô hình BERT
```bash
python train_ner.py
```

### Chạy thử nghiệm Pipeline trích xuất thực thể (Information Extraction)
```bash
python information_extraction.py
```

### Xem trực quan trên Jupyter Notebook
Mở và chạy tuần tự các cell trong:
- `tokenizer_alignment.ipynb` (Khảo sát căn chỉnh nhãn subword)
- `ner_finetuning.ipynb` (Quy trình huấn luyện, biểu đồ trực quan hóa và ứng dụng)
