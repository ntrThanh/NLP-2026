# Báo cáo Chiêm nghiệm & Trả lời Vấn đáp (Reflection & Learning Check) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** From Static Representations to Sequential Modeling (RNN, LSTM, GRU)

---

## 1. Mạch nối kiến trúc xuyên suốt: LAB 01 $\to$ LAB 05 $\to$ LAB 06/07

Chương trình môn học được thiết kế theo một lộ trình tiến hóa tự nhiên và chặt chẽ của các kỹ thuật biểu diễn ngôn ngữ trong NLP:

```
LAB 01: Văn bản ──> TF-IDF ──> Biểu diễn thưa (Sparse Lexical) ──> Tìm kiếm thông tin
                                      │
LAB 02: Văn bản ──> N-gram ──> Mô hình xác suất chuỗi ──> Language Model (Perplexity)
                                      │
LAB 03: Văn bản ──> Word2Vec ──> Biểu diễn dày (Dense Semantic) ──> Tương đồng ngữ nghĩa
                                      │
LAB 04: Biểu diễn ──> Phân loại có giám sát (Classifier) ──> Đánh giá & Phân tích lỗi
                                      │
LAB 05: Chuỗi từ ──> Mạng hồi quy (RNN/LSTM/GRU) ──> Trạng thái ẩn tuần tự (Sequential State)
                                      │
LAB 06 & 07: Trực tiếp kết nối ngữ cảnh ──> Cơ chế Attention & Transformer (Không cần hồi quy)
```

### Bài học cốt lõi rút ra:
1. **Ở LAB 04, ta nhận ra giới hạn lớn của Biểu diễn tĩnh (Static Representations):**
   - Cả TF-IDF (Bag-of-Words) và Average Word2Vec đều biểu diễn toàn bộ văn bản bằng **một vector cố định duy nhất ngay từ đầu**.
   - Phép cộng giao hoán của túi từ làm mất hoàn toàn trật tự từ: Câu *"the movie was good, not bad"* và *"the movie was bad, not good"* có cùng vector TF-IDF, khiến mô hình hoàn toàn bất lực trước hiện tượng phủ định hay đảo nghĩa.
2. **LAB 05 giải quyết bài toán trật tự từ bằng Biểu diễn tuần tự (Sequential Modeling):**
   - Thay vì nén toàn văn bản thành một vector cố định, mạng đọc từng token theo thời gian $t = 1, 2, \dots, T$.
   - Mỗi bước cập nhật trạng thái ẩn: $h_t = f(x_t, h_{t-1})$. Trạng thái ẩn $h_t$ đóng vai trò như một "bộ nhớ ngữ cảnh" tích lũy thông tin của toàn bộ các từ đứng trước nó.
3. **Nhưng Recurrent Modeling bộc lộ điểm nghẽn nghiêm trọng dẫn tới LAB 06 (Attention):**
   - Quá trình tính toán hồi quy mang tính tuần tự bắt buộc: Muốn tính trạng thái tại từ thứ 100 ($x_{100}$), mô hình bắt buộc phải chạy qua 99 bước trước đó. Điều này **không thể tính toán song song (parallelization)** trên GPU.
   - Khi chuỗi dài, dù LSTM và GRU có các cổng kiểm soát thì việc ép toàn bộ thông tin của hàng trăm từ vào một vector ẩn có kích thước hữu hạn (vector bottleneck) vẫn làm thất thoát ngữ cảnh chi tiết.
   - **Động lực ra đời của LAB 06 (Attention):** Thay vì bắt thông tin truyền qua chuỗi hồi quy dài, cơ chế Attention cho phép token ở bước $t$ có thể "nhìn thẳng" và gán trọng số trực tiếp tới mọi token quan trọng khác trong câu, mở đường cho kỷ nguyên Transformer.

---

## 2. Bảng so sánh thuộc tính toàn diện giữa các thế hệ mô hình

| Thuộc tính (Property) | TF-IDF + Logistic Regression | Average Word2Vec + LR | Simple RNN | LSTM | GRU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Không gian biểu diễn** | Sparse (Thưa, $20,000$ chiều) | Dense (Dày, $100-300$ chiều) | Dense (Dày, nhúng học được) | Dense (Dày, nhúng học được) | Dense (Dày, nhúng học được) |
| **Mô hình hóa trật tự từ** | Không (Mất hoàn toàn) | Không (Phép cộng giao hoán) | **Có** (Tuần tự từng bước) | **Có** (Tuần tự từng bước) | **Có** (Tuần tự từng bước) |
| **Trạng thái phụ thuộc ngữ cảnh** | Không | Không | **Có** ($h_t$) | **Có** ($h_t, c_t$) | **Có** ($h_t$) |
| **Xử lý phụ thuộc xa ($T > 100$)** | Không có khái niệm chuỗi | Không có khái niệm chuỗi | **Rất yếu** (Vanishing Gradient) | **Tốt** (Băng chuyền Cell state) | **Tốt** (Cổng Update & Reset) |
| **Khả năng giải thích (Interpretability)** | Rất cao (Hệ số trọng số $w_i$) | Trung bình (Không gian vector) | Thấp (Hộp đen phi tuyến) | Thấp (Nhiều cổng phi tuyến) | Thấp (Nhiều cổng phi tuyến) |
| **Chi phí tính toán (Training cost)** | Rất thấp (~12s trên CPU) | Rất thấp (~3-5s) | Thấp (~43s / 10 epochs) | Trung bình (~80s / 10 epochs) | Khá (~119s / 10 epochs) |
| **F1-Score thực nghiệm (IMDb Test)** | 0.9036 (Nhờ từ vựng mạnh) | 0.8449 (Loãng do trung bình) | 0.6559 (Mắc kẹt gradient) | 0.7944 (Cải thiện rõ rệt) | **0.8409** (Tối ưu chuỗi tốt nhất) |

---

## 3. Trả lời các câu hỏi kiểm tra vấn đáp cá nhân (Individual Learning Check)

### Câu 1: Tại sao Accuracy 95% chưa chắc là một mô hình tốt?
**Trả lời:**
- Khi tập dữ liệu bị mất cân bằng lớp trầm trọng (Class Imbalance) — ví dụ tập dữ liệu y tế phát hiện bệnh hiếm có 950 mẫu âm tính và 50 mẫu dương tính (tỷ lệ 95:5).
- Một mô hình "ngây thơ" chỉ cần dự đoán tất cả mẫu là Âm tính sẽ nghiễm nhiên đạt Accuracy là $950/1000 = 95\%$. Tuy nhiên mô hình này có **Recall = 0%** trên lớp dương tính, hoàn toàn vô dụng trong thực tế. Vì vậy, bắt buộc phải xem xét thêm **Precision, Recall, F1-Score** và **PR-AUC**.

### Câu 2: TF-IDF khác Average Word2Vec ở những điểm bản chất nào?
**Trả lời:**
- **TF-IDF:** Là biểu diễn thưa (sparse, lexical). Tọa độ của vector tương ứng với sự hiện diện và độ quan trọng thống kê của từng từ vựng cụ thể. Bắt rất tốt các từ khóa mang cảm xúc mạnh (*excellent, awful*) nhưng hoàn toàn không biết hai từ đồng nghĩa (*great* và *superb*) có liên quan nhau.
- **Average Word2Vec:** Là biểu diễn dày (dense, semantic). Mỗi từ được biểu diễn bằng vector ngữ nghĩa liên tục. Việc cộng trung bình giúp bảo toàn sự tương đồng ngữ nghĩa phân bố, nhưng lại làm "loãng" các từ khóa cảm xúc quyết định trong một biển các từ trung tính.

### Câu 3: Tại sao cụm từ "not good" có thể gây lỗi nghiêm trọng cho mô hình Bag-of-Words?
**Trả lời:**
- Mô hình Bag-of-Words (Unigram TF-IDF hoặc Count) xem văn bản như một tập hợp từ rời rạc: $\{ \text{"not"}: 1, \text{"good"}: 1 \}$.
- Do trọng số dương của từ *"good"* trong các bộ phân loại thường rất lớn ($w_{\text{good}} \gg 0$), trong khi từ *"not"* là hư từ xuất hiện ở cả hai lớp, tổng tuyến tính $w^T x$ vẫn dương và mô hình đoán nhầm là Positive. Mô hình chỉ sửa được lỗi này khi mở rộng lên **Bigram** (tạo đặc trưng riêng `"not_good"` có trọng số âm lớn) hoặc dùng **Sequence Model** (RNN/LSTM/GRU).

### Câu 4: Data Leakage xảy ra ở đâu nếu thực hiện fit TF-IDF trước khi chia Train/Test?
**Trả lời:**
- Nếu fit `TfidfVectorizer` trên toàn bộ tập dữ liệu trước khi split, **từ điển vựng (vocabulary)** và **tần số nghịch đảo văn bản (IDF)** đã được tính toán dựa trên cả các từ xuất hiện trong tập Test.
- Khi đó, thông tin về tần suất và sự phân bố của dữ liệu kiểm thử trong tương lai đã bị rò rỉ vào quá trình học đặc trưng của tập Train. Đánh giá kiểm thử sẽ bị lạc quan quá mức (over-optimistic) và không phản ánh đúng năng lực tổng quát hóa thực tế.

### Câu 5: Precision và Recall khác nhau thế nào và khi nào ưu tiên chỉ số nào?
**Trả lời:**
- $\text{Precision} = \frac{TP}{TP + FP}$: Đo lường trong số các mẫu mô hình đoán là Positive, có bao nhiêu mẫu thực sự là Positive (độ chính xác của cảnh báo).
- $\text{Recall} = \frac{TP}{TP + FN}$: Đo lường trong số các mẫu thực sự là Positive, mô hình đã tóm được bao nhiêu mẫu (độ bao phủ).
- **Ứng dụng:**
  - Ưu tiên **Precision** khi chi phí của False Positive rất đắt (ví dụ: bộ lọc Spam gửi nhầm email quan trọng của khách hàng vào thùng rác).
  - Ưu tiên **Recall** khi chi phí của False Negative rất nguy hiểm (ví dụ: chẩn đoán ung thư, phát hiện giao dịch gian lận ngân hàng).

### Câu 6: Tại sao Simple RNN bị triệt tiêu đạo hàm (Vanishing Gradient) nhưng LSTM lại khắc phục được?
**Trả lời:**
- **Trong Simple RNN:** Đạo hàm lan truyền ngược qua thời gian (BPTT) từ bước $T$ về bước $t$ là tích của một chuỗi Jacobian:
  $$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \operatorname{diag}(1 - h_k^2) W_{hh}^T$$
  Vì $\tanh'(x) = 1 - h_k^2 \le 1$, việc nhân liên tiếp hàng chục ma trận có chuẩn nhỏ hơn 1 khiến gradient suy giảm theo hàm mũ về $0$.
- **Trong LSTM:** Thông tin ô nhớ $c_t$ được cập nhật theo cơ chế **cộng tuyến tính (Additive Update)**:
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \implies \frac{\partial c_t}{\partial c_{t-1}} = f_t$$
  Nếu cổng quên $f_t \approx 1$, gradient có thể truyền thẳng ngược về quá khứ qua hàng trăm bước mà không bị nhân với ma trận trọng số $W$ hay bị ép qua hàm phi tuyến co hẹp. Đây chính là "đại lộ gradient" (gradient highway) giúp LSTM học được các phụ thuộc cực xa.

---

## 4. Tuyên bố sử dụng AI (AI Assistance Statement)

Tuân thủ nghiêm ngặt theo quy định học thuật tại **Mục 24 của đề cương LAB 05 (W5.pdf)**:

- **Công cụ hỗ trợ (Tool):** Google Antigravity (DeepMind Advanced Agentic AI Assistant).
- **Mục đích sử dụng (Purpose):** Hỗ trợ viết boilerplate mã nguồn PyTorch, trực quan hóa biểu đồ Matplotlib, định dạng bảng biểu và cấu trúc hóa báo cáo học thuật.
- **Phần do AI hỗ trợ sinh (Generated):**
  - Khung lớp mô hình `SimpleRNNCell`, `SequenceClassifier` và vòng lặp huấn luyện PyTorch.
  - Các hàm vẽ đồ thị Learning Curves (`learning_curves.png`) và ma trận nhầm lẫn (`confusion_matrix.png`).
- **Phần sinh viên tự điều chỉnh & tùy biến (Modified):**
  - Tinh chỉnh các siêu tham số huấn luyện (`embed_dim=64`, `hidden_dim=64`, `lr=0.001`, `max_norm=5.0`, `epochs=10`).
  - Thiết kế các kịch bản kiểm thử độc lập cho Experiment 5 (Sequence length trade-off) và Experiment 6 (Long-range dependency vị trí từ khóa).
- **Quy trình kiểm chứng độc lập (Verified):**
  - Tự tay chạy và giám sát quá trình huấn luyện từng epoch trên kernel Python Jupyter Notebook.
  - Kiểm định đối chiếu lớp `SimpleRNNCell` với `torch.nn.RNNCell` qua unit test số học sai số $< 10^{-6}$.
  - Trực tiếp đối chiếu các mẫu phân tích lỗi và số liệu trong bảng `results.csv` với dữ liệu thực tế của tập Test IMDb.

