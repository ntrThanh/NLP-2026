# LAB 05 — Sequence Models: RNN, LSTM and GRU

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** From Static Representations to Sequential Modeling  

---

## 1. Giới thiệu tổng quan & Mục tiêu học tập

Ở **LAB 04**, các kỹ thuật biểu diễn tĩnh như **TF-IDF** (Bag-of-Words) và **Average Word2Vec** đã bộc lộ giới hạn cốt lõi: chúng biểu diễn văn bản bằng một vector cố định ngay từ đầu, làm mất hoàn toàn thông tin về trật tự từ và phạm vi phủ định (negation scope).

**LAB 05** đánh dấu bước chuyển mình quan trọng sang phương pháp tiếp cận tuần tự (**Sequential Modeling**):
$$\text{Raw Text} \longrightarrow \text{Token Sequence} \longrightarrow \text{Learned Embedding} \longrightarrow \text{RNN / LSTM / GRU} \longrightarrow \text{Hidden State } h_T \longrightarrow \text{Classifier}$$

Mô hình đọc từng từ theo dòng thời gian và liên tục cập nhật trạng thái ẩn:
$$h_t = f(x_t, h_{t-1})$$
Từ đó, mô hình duy trì ngữ cảnh động xuyên suốt chuỗi, khắc phục nhược điểm của túi từ tĩnh và làm bàn đạp trực tiếp hướng tới cơ chế **Attention** và **Transformer** (ở LAB 06 & 07).

---

## 2. Cấu trúc thư mục Lab 05 (Deliverables)

```text
lab05/
├── README.md                          # Báo cáo tổng quan toàn bộ Lab 05 và hướng dẫn thực thi
├── rnn_cell.py                        # Module tự cài đặt SimpleRNNCell từ đầu (Mục 14)
├── experiments.ipynb                  # Notebook thực nghiệm toàn diện từ Exp 1 đến Exp 6
├── results.csv                        # Bảng tổng hợp số liệu thực nghiệm của tất cả mô hình
├── sequence_length_histogram.png      # Phân phối độ dài chuỗi tập dữ liệu IMDb (Exp 1)
├── padding_truncation_experiment.png  # Khảo sát ảnh hưởng của Padding và Truncation (Exp 2)
├── rnn_lstm_gru_comparison.png        # Biểu đồ so sánh hiệu năng và thời gian RNN, LSTM, GRU (Exp 4)
├── learning_curves.png                # Đồ thị Train Loss và Validation Loss qua 10 Epochs (Exp 4)
├── sequence_length_tradeoff.png       # Khảo sát trade-off giữa ngữ cảnh và chi phí tính toán (Exp 5)
├── confusion_matrix.png               # Ma trận nhầm lẫn của mô hình GRU tốt nhất trên tập Test (Mục 24)
├── error_analysis.md                  # Phân tích chuyên sâu 6 mẫu lỗi thực tế từ tập Test
└── reflection.md                      # Báo cáo chiêm nghiệm, trả lời vấn đáp và tuyên bố AI policy
```

---

## 3. Bảng kết quả định lượng tổng hợp (`results.csv`)

Toàn bộ các mô hình được huấn luyện và đánh giá trên tập dữ liệu chuẩn **IMDb Movie Reviews** (Train: 34,707 mẫu, Validation: 7,437 mẫu, Test: 7,438 mẫu) với cùng từ điển 20,000 từ vựng và hạt giống ngẫu nhiên cố định (`seed=42`):

| Mô hình (Model) | Biểu diễn (Representation) | Số tham số (Parameters) | Thời gian huấn luyện (s) | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **TF-IDF** (Sparse Lexical) | 20,001 | 12.03s | **0.9024** | **0.8958** | **0.9116** | **0.9036** |
| **Simple RNN** | **Learned Embedding** | 1,288,513 | 43.96s (10 epochs) | 0.6262 | 0.6096 | 0.7099 | 0.6559 |
| **LSTM** | **Learned Embedding** | 1,313,473 | 76.77s (10 epochs) | 0.8038 | 0.8380 | 0.7552 | 0.7944 |
| **GRU** | **Learned Embedding** | 1,305,153 | 114.09s (10 epochs) | **0.8330** | **0.8058** | **0.8792** | **0.8409** |

---

## 4. Tóm tắt các thực nghiệm cốt lõi

### 4.1. Experiment 1: Dataset Inspection & Phân phối độ dài chuỗi
- **Kích thước dữ liệu:** Cân bằng hoàn hảo 50:50 (Train: 17,419 pos / 17,288 neg). Không có nhãn thiếu và 0 mẫu trùng lặp giữa các tập (không có rò rỉ Data Leakage).
- **Phân phối độ dài:** Min: 4 từ, Median: 173 từ, Mean: 231.72 từ, Max: 2,470 từ. Phân vị $p_{75} = 280$, $p_{85} = 373$, $p_{90} = 454$, $p_{95} = 594$ từ.
- **Tiêu chí chọn `max_length`:** Phải cân bằng giữa tỷ lệ giữ thông tin ($p_{80}-p_{90}$) và chi phí tính toán $\mathcal{O}(T)$ cùng hiện tượng nhiễu từ token `<PAD>`.

### 4.2. Experiment 2: Padding & Truncation
- Khi tăng `max_length` từ $50 \to 100 \to 200$, lượng từ giữ lại tăng từ $21.49\% \to 41.51\% \to 68.57\%$.
- Với Simple RNN, tăng độ dài chuỗi **không làm tăng hiệu năng** mà ngược lại khiến mô hình khó học hơn do hiện tượng suy giảm gradient theo cấp số nhân qua thời gian.

### 4.3. Experiment 3: Baseline & Bằng chứng thực nghiệm về trật tự từ
- **Thực nghiệm cặp câu hoán vị trật tự:** Xét *"the movie was good, not bad"* vs *"the movie was bad, not good"*.
  - TF-IDF hoàn toàn bất lực: $\Delta P(\text{Positive}) = 0.000000$ (cho ra cùng xác suất $0.0070$).
  - Simple RNN nhận diện rõ sự khác biệt của từ cuối câu: $\Delta P(\text{Positive}) > 0.03$.
- **Thực nghiệm xáo trộn từ (Word Shuffle Permutation):** Khi xáo trộn ngẫu nhiên toàn bộ từ trong câu test, TF-IDF giữ nguyên 100% độ chính xác (mù trật tự), trong khi độ chính xác của mạng hồi quy bị sụt giảm ngay lập tức.

### 4.4. Experiment 4: So sánh RNN vs LSTM vs GRU (10 Epochs)
- **Simple RNN (F1 = 0.6559):** Bị hạn chế bởi triệt tiêu đạo hàm (Vanishing Gradient qua chuỗi nhân Jacobian).
- **LSTM (F1 = 0.7944):** Cải thiện vượt bậc nhờ cơ chế **Cell State $c_t$** hoạt động như một băng chuyền thông tin tuyến tính nối tắt (gradient highway: $\frac{\partial c_t}{\partial c_{t-1}} = f_t$).
- **GRU (F1 = 0.8409):** Đạt điểm F1 cao nhất toàn bộ các mạng nơ-ron tuần tự. Với cấu trúc 2 cổng tinh gọn (Reset $r_t$ và Update $z_t$), GRU hội tụ nhanh, kiểm soát luồng thông tin dài hạn xuất sắc trên tập dữ liệu đánh giá phim.

### 4.5. Experiment 5: Sequence Length Trade-off trên GRU
- Khảo sát GRU với `max_length` $\in \{50, 100, 200\}$.
- Ngưỡng `max_length = 100` là điểm cân bằng tối ưu (Sweet spot): đạt F1 ~0.84 với thời gian huấn luyện chỉ ~1.5 phút trên CPU, trong khi tăng lên 200 làm tăng gấp đôi thời gian nhưng F1 có xu hướng bão hòa.

### 4.6. Experiment 6: Long-range Dependency
- Kiểm chứng độ nhạy với từ khóa cảm xúc ở các vị trí: **Đầu câu**, **Giữa câu**, **Cuối câu**.
- Simple RNN bị suy giảm độ nhạy mạnh khi từ khóa xuất hiện ở đầu câu dài. Ngược lại, GRU giữ vững độ nhạy đồng đều trên mọi vị trí nhờ cổng Update gate không làm suy hao tín hiệu ẩn.

---

## 5. Hướng dẫn cài đặt & Tái lập kết quả (Reproduction Guide)

### Yêu cầu hệ thống:
- Hệ điều hành: Linux (Ubuntu 22.04+)
- Trình thông dịch Python: Python 3.10+ (Đã thử nghiệm thành công trên môi trường conda `ml` với PyTorch 2.14 CPU).
- Phần cứng: CPU đa nhân thông thường (ví dụ AMD Ryzen 7 7730U), không yêu cầu GPU rời.

### Các bước thực thi:

1. **Kiểm tra unit test của module SimpleRNNCell:**
   ```bash
   python3 lab05/rnn_cell.py
   ```
   *Kỳ vọng:* `Verification status: True, max absolute diff: < 1e-6`.

2. **Chạy toàn bộ thí nghiệm:**
   Mở và chạy lần lượt các cell trong notebook [`lab05/experiments.ipynb`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab05/experiments.ipynb). Toàn bộ quá trình từ Experiment 1 đến Experiment 6 sẽ hoàn tất sau khoảng 5-6 phút trên CPU.

3. **Kiểm tra các sản phẩm đầu ra đã sinh:**
   - File số liệu: [`lab05/results.csv`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab05/results.csv)
   - Đồ thị học tập: [`lab05/learning_curves.png`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab05/learning_curves.png)
   - Ma trận nhầm lẫn: [`lab05/confusion_matrix.png`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab05/confusion_matrix.png)
   - Báo cáo chi tiết: [`lab05/error_analysis.md`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab05/error_analysis.md) và [`lab05/reflection.md`](file:///media/trong-thanh/Data/University/Fourth%20Year/NLP/lab05/reflection.md).

