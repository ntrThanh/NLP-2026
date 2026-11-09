# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** From Training a Transformer to Using a Pretrained Language Model  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  

---

## 1. Giới thiệu tổng quan & Mạch kiến thức

Ở **LAB 07**, chúng ta đã tự tay xây dựng kiến trúc Transformer Encoder từ các tầng cơ bản (Self-Attention, MHA, Residual Connection, LayerNorm, FFN) và huấn luyện trên bài toán cụ thể. Tuy nhiên, mô hình phải học biểu diễn ngôn ngữ từ con số 0 (*from scratch*).

**LAB 08** đặt câu hỏi nền tảng của NLP hiện đại:
> *Nếu một Transformer đã được huấn luyện trước trên một lượng văn bản rất lớn (Pre-training) thì có thể sử dụng kiến thức đó cho một bài toán mới như thế nào (Fine-tuning)?*

### Mạch chuyển tiếp học phần:
$$\begin{matrix}
\text{LAB 07: Transformer from scratch} \\
\downarrow \\
\textbf{LAB 08: Pretraining} \to \textbf{Masked Language Modeling (MLM)} \to \textbf{BERT} \to \textbf{Fine-tuning} \to \textbf{Text Classification} \\
\downarrow \\
\text{LAB 09: Token Classification / Named Entity Recognition (NER)}
\end{matrix}$$

---

## 2. Cấu trúc thư mục nộp bài (Deliverables Checklist)

Thư mục `lab08/` được tổ chức đầy đủ và chuẩn xác theo quy định tại Mục 25 của đề bài:

```text
lab08/
├── README.md                          # Báo cáo tổng quan, số liệu thực nghiệm và tính tái lập (Reproducibility)
├── W8.pdf                             # Đề bài và quy định thực hiện Lab 08
├── calculations.md                    # Lời giải chi tiết Calculation Exercise 1 (MLM Loss) & 2 (Tokenization)
├── calculations.pdf                   # Bản in định dạng PDF của báo cáo tính toán lý thuyết
├── prediction.md                      # 3 dự đoán lý thuyết trước thực nghiệm và bảng đối chiếu kiểm chứng (Mục 11)
├── tokenization.ipynb                 # Notebook thực thi minh họa Tokenization & Special tokens (Mục 10 & 12)
├── pretrained_classification.py       # Module mã nguồn Python cài đặt PretrainedClassifier pipeline (Mục 13)
├── experiments.ipynb                  # Notebook thực nghiệm toàn diện các Experiment 1 -> 5 (Google Colab / VS Code)
├── fine_tuning.ipynb                  # Bản sao notebook Fine-tuning theo quy chuẩn danh mục đề bài
├── extract_outputs.py                 # Script tự động trích xuất kết quả và biểu đồ từ notebook
├── results.csv                        # Bảng tổng hợp số liệu định lượng chi tiết của toàn bộ 5 thí nghiệm
├── error_samples.csv                  # Dữ liệu trích xuất các mẫu dự đoán sai thực tế từ tập Test IMDb
├── learning_curves.png                # Đồ thị đường cong huấn luyện Train Loss và Val Loss qua các Epochs
├── confusion_matrix.png               # Ma trận nhầm lẫn của mô hình BERT Fine-Tuning trên tập Test IMDb
├── data_efficiency.png                # Đồ thị hiệu quả dữ liệu: So sánh Pretrained vs From-scratch (10% - 100%)
├── confidence_analysis.png            # Biểu đồ phân phối độ tin cậy và minh chứng Confidence != Correctness
├── error_analysis.md                  # Phân tích định tính 8 mẫu lỗi thực tế và trả lời 6 câu hỏi bắt buộc (Mục 20 & 28)
└── reflection.md                      # Bài luận phản tư 300-500 từ về lý do không train Transformer from scratch (Mục 29)
```

---

## 3. Tính Tái lập Thực nghiệm (Reproducibility — Mục 23)

Để đảm bảo tính minh bạch khoa học và khả năng tái lập kết quả, toàn bộ tham số môi trường và siêu tham số huấn luyện được công bố chi tiết:

* **Mô hình nền tảng (Backbone):** `bert-base-uncased` (12 layers, 768 hidden, 12 heads, 110M params)
* **Bộ tách từ (Tokenizer):** WordPiece Tokenizer (`AutoTokenizer.from_pretrained("bert-base-uncased")`)
* **Tập dữ liệu (Dataset):** IMDb Large Movie Review Dataset (gồm 2 nhãn: *Negative* và *Positive*)
* **Phân chia dữ liệu (Splits):**
  - Train: 4,000 mẫu (được cân bằng 2 lớp)
  - Validation: 1,000 mẫu
  - Test: 1,000 mẫu
* **Độ dài chuỗi tối đa (Max sequence length):** $T = 128$ (mặc định)
* **Kích thước Batch (Batch size):** $16$
* **Tốc độ học (Learning rate):** $2 \times 10^{-5}$ với Linear Warmup và Decay
* **Số Epochs:** $3$ epochs
* **Thuật toán tối ưu (Optimizer):** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, Weight Decay = $0.01$)
* **Hạt giống ngẫu nhiên (Random seed):** $42$
* **Môi trường phần cứng (Hardware):** GPU Tesla T4 (Google Colab) / Intel Core i7 CPU (Local)
* **Thời gian huấn luyện trung bình:** $\approx 178.34$ giây cho Full Fine-tuning

---

## 4. Tổng hợp Kết quả Thực nghiệm (Experimental Results)

Dưới đây là bảng số liệu chi tiết được trích xuất trực tiếp từ [results.csv](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab08/results.csv):

| Thí nghiệm | Thiết lập (Setting) | Khởi tạo (Init) | Encoder Trainable? | Tập Train | Max Len | Accuracy | Precision | Recall | Macro F1 | Thời gian (s) | # Tham số tối ưu |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1** | Transformer from scratch | Random | Yes | 4,000 | 128 | **0.4980** | 0.0000 | 0.0000 | 0.0000 | 170.19s | 108.89M |
| **Exp 1** | **Pretrained BERT Fine-tuning** | **Pretrained** | **Yes** | **4,000** | **128** | **0.8600** | **0.8390** | **0.8924** | **0.8649** | **178.34s** | **108.89M** |
| **Exp 2** | Frozen Encoder (Feature Extraction) | Pretrained | No | 4,000 | 128 | **0.7790** | 0.7607 | 0.8167 | 0.7877 | **65.68s** | **1,538** |
| **Exp 3** | Pretrained BERT (10% Data) | Pretrained | Yes | 400 | 128 | **0.8260** | 0.8333 | 0.8167 | 0.8249 | 29.34s | 108.89M |
| **Exp 3** | From Scratch (10% Data) | Random | Yes | 400 | 128 | **0.5020** | 0.5020 | 1.0000 | 0.6684 | 29.32s | 108.89M |
| **Exp 3** | Pretrained BERT (25% Data) | Pretrained | Yes | 1,000 | 128 | **0.8440** | 0.8327 | 0.8625 | 0.8474 | 54.28s | 108.89M |
| **Exp 3** | From Scratch (25% Data) | Random | Yes | 1,000 | 128 | **0.5020** | 0.5020 | 1.0000 | 0.6684 | 53.97s | 108.89M |
| **Exp 3** | Pretrained BERT (50% Data) | Pretrained | Yes | 2,000 | 128 | **0.8470** | 0.8483 | 0.8466 | 0.8475 | 95.59s | 108.89M |
| **Exp 3** | From Scratch (50% Data) | Random | Yes | 2,000 | 128 | **0.5020** | 0.5020 | 1.0000 | 0.6684 | 95.06s | 108.89M |
| **Exp 4** | Sequence Length $T = 64$ | Pretrained | Yes | 4,000 | 64 | **0.8230** | 0.8095 | 0.8466 | 0.8277 | 94.59s | 108.89M |
| **Exp 4** | Sequence Length $T = 256$ | Pretrained | Yes | 4,000 | 256 | **0.9030** | 0.8917 | 0.9183 | 0.9048 | 370.93s | 108.89M |
| **Exp 5** | In-domain (IMDb $\to$ IMDb) | Pretrained | Yes | 4,000 | 128 | **0.8600** | 0.8390 | 0.8924 | 0.8649 | — | — |
| **Exp 5** | Out-of-domain (IMDb $\to$ Amazon Reviews) | Pretrained | Yes | 4,000 | 128 | **0.9040** | 0.8621 | 0.9690 | 0.9124 | — | — |

---

## 5. Các phát hiện cốt lõi từ Thực nghiệm

1. **Pretrained vs. From-scratch (Exp 1):** Mô hình Transformer train from scratch hoàn toàn thất bại khi chỉ có 4,000 mẫu dữ liệu ($49.8\%$), trong khi Pretrained BERT đạt $86.0\%$. Điều này khẳng định vai trò sống còn của giai đoạn Pre-training trong việc cung cấp biểu diễn ngữ nghĩa tổng quát.
2. **Frozen vs. Full Fine-tuning (Exp 2):** Đóng băng encoder chỉ huấn luyện 1,538 tham số nhưng vẫn đạt $77.9\%$ Accuracy trong 65.68 giây (nhanh gấp gần 3 lần). Tuy nhiên, Full Fine-tuning giúp tăng thêm $+8.1\%$ độ chính xác nhờ cơ chế tái thích ứng các đầu chú ý với miền cảm xúc.
3. **Hiệu quả dữ liệu (Data Efficiency - Exp 3):** Pretrained BERT với chỉ 10% dữ liệu (400 mẫu) vẫn đạt độ chính xác $82.6\%$, cao hơn áp đảo mô hình from-scratch học trên 100% dữ liệu.
4. **Ảnh hưởng độ dài chuỗi (Exp 4):** Tăng $T=64 \to 128 \to 256$ giúp Accuracy tăng từ $82.3\% \to 86.0\% \to 90.3\%$, nhưng thời gian huấn luyện tăng gấp 4 lần ($94.59\text{s} \to 370.93\text{s}$) và bộ nhớ tiêu thụ tăng từ $4.7\text{GB} \to 8.5\text{GB}$, phản ánh độ phức tạp toán học $O(T^2)$ của Self-Attention.
5. **Cảnh báo phương pháp luận (Confidence != Correctness):** Phân tích 8 mẫu lỗi điển hình trong `error_analysis.md` chỉ ra rằng mô hình rất dễ bị "đánh lừa" bởi các từ ngữ bề mặt trong câu trào phúng (sarcasm) hoặc các review dài bị cắt ngắn đoạn đầu do `max_length`, dẫn đến các dự đoán sai lệch với độ tự tin cực đoan ($P > 99.5\%$).

---

## 6. Hướng dẫn chạy lại mã nguồn (How to Run)

### Kích hoạt môi trường:
```bash
conda activate ml
```

### Chạy kiểm tra kiến trúc và pipeline phân loại:
```bash
python lab08/pretrained_classification.py
```

### Chạy notebook thực nghiệm:
Mở và thực thi các cell trong [experiments.ipynb](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab08/experiments.ipynb) hoặc [tokenization.ipynb](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab08/tokenization.ipynb) thông qua Jupyter Notebook hoặc VS Code.

---

## 7. Khai báo Sử dụng AI (AI Usage Declaration — Mục 24)

Tuân thủ quy định tại Mục 24 của tài liệu thực hành Lab 08:

* **AI-assisted code:** Hỗ trợ chuẩn hóa cấu trúc boilerplate pipeline phân loại văn bản và định dạng biểu đồ trực quan hóa kết quả.
* **What was generated:** Khung sườn mã nguồn `pretrained_classification.py` và script tạo notebook `tokenization.ipynb`.
* **What was modified:** Tinh chỉnh cơ chế trích xuất vector `[CLS]` ở tầng cuối cùng, bổ sung Dropout, và thiết lập chế độ đóng băng trọng số encoder (`requires_grad = False`) cho thí nghiệm Feature Extraction.
* **What was verified:** Kiểm tra toàn bộ kích thước tensor đầu ra, tính đúng đắn của attention mask khi padding, kiểm chứng giá trị hàm mất mát Cross-Entropy và xác minh tính nhất quán của bảng số liệu thực nghiệm so với lý thuyết.

