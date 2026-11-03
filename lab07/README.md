# LAB 07 — Transformer Architecture: From Attention to Transformer

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** From Attention to Transformer  

---

## 1. Giới thiệu tổng quan & Mục tiêu học tập

Ở **LAB 05**, sinh viên đã làm việc với các mô hình hồi quy tuần tự (RNN, LSTM, GRU). Ở **LAB 06**, cơ chế **Attention** và **Self-Attention** đã được giới thiệu để kết nối trực tiếp các vị trí trong chuỗi.

**LAB 07** trả lời câu hỏi cốt lõi:
> *Làm thế nào để kết hợp Self-Attention, Positional Encoding, Feed-Forward Network, Residual Connection và Layer Normalization thành một kiến trúc Transformer Encoder hoàn chỉnh phục vụ biểu diễn và phân loại văn bản?*

### Các mục tiêu học tập (Learning Outcomes):
- **LO1 - LO4:** Hiểu và tính toán Self-Attention, Multi-Head Attention, Positional Encoding.
- **LO5 - LO6:** Phân tích vai trò của Position-wise Feed-Forward Network (FFN), Residual Connection và LayerNorm.
- **LO7 - LO8:** Tự cài đặt khối `TransformerBlock` nguyên thủy từ các tầng cơ bản (không dùng sẵn `TransformerEncoderLayer`).
- **LO9 - LO12:** So sánh thực nghiệm Transformer vs LSTM, đánh giá ảnh hưởng của số heads, chiến lược pooling, và chi phí bậc hai $O(T^2)$.
- **LO13:** Chuẩn bị nền tảng chuyển tiếp vững chắc sang **LAB 08 (Pretrained Transformer / BERT)**.

---

## 2. Cấu trúc thư mục Lab 07 (Deliverables)

```text
lab07/
├── README.md                          # Báo cáo tổng quan toàn bộ Lab 07 và hướng dẫn thực thi
├── prediction.md                      # 6 dự đoán trước thực nghiệm và bảng đối chiếu kiểm chứng (Mục 8 & 28)
├── transformer_block.py               # Module mã nguồn Python tự cài đặt Transformer Encoder Block (Mục 17 & 18)
├── positional_encoding.py             # Module cài đặt & thực nghiệm kiểm chứng Positional Encoding (Mục 11 & 12)
├── experiments.ipynb                  # Notebook thực nghiệm toàn diện từ Exp 1 đến Exp 6 (đã chạy có output)
├── results.csv                        # Bảng tổng hợp số liệu thực nghiệm LSTM vs Transformer
├── positional_encoding.png            # Biểu đồ heatmap PE và khả năng phân biệt câu hoán vị (Exp 1 & 12)
├── attention_heads.png                # Heatmap ma trận trọng số chú ý đa dạng qua 4 Attention Heads (Exp 3)
├── learning_curves.png                # Đồ thị Train Loss và Validation Loss qua các Epochs (Exp 4)
├── confusion_matrix.png               # Ma trận nhầm lẫn của mô hình Transformer trên tập Test IMDb
├── error_analysis.md                  # Phân tích chuyên sâu 6 mẫu lỗi thực tế và cảnh báo phương pháp luận
└── reflection.md                      # Báo cáo chiêm nghiệm, bảng so sánh Lab 01-07 và trả lời vấn đáp
```

---

## 3. Bảng Kết quả Thực nghiệm Tổng hợp (`results.csv`)

Mô hình được huấn luyện và đánh giá trên tập dữ liệu chuẩn **IMDb Movie Reviews** với cùng từ điển 20,000 từ vựng và hạt giống ngẫu nhiên cố định (`seed=42`):

| Mô hình (Model) | Kiểu biểu diễn (Representation) | Số tham số | Thời gian train (s) | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Baseline)** | Sequential Hidden State $h_T$ | 2,692,481 | 30.01s | 0.6124 | 0.6325 | 0.5218 | 0.5718 |
| **Transformer (4 Heads)** | Multi-Head Self-Attention + FFN | 2,692,353 | 121.64s | **0.7444** | **0.7795** | **0.6758** | **0.7240** |

---

## 4. Tóm tắt các Phát hiện Thực nghiệm Chính

1. **Positional Encoding là bắt buộc:**
   - Không có PE, khoảng cách biểu diễn giữa *"dog bites man"* và *"man bites dog"* bằng $0.0000$ (trùng nhau 100% do tính chất *permutation equivariance*).
   - Có PE, khoảng cách tăng lên $0.2164$, giúp Transformer nhận biết chính xác trật tự trước sau.
2. **Hiệu quả của Multi-Head:**
   - Khi tăng $n_{\text{heads}}$ từ 1 lên 8 (với $d_{\text{model}}=128$ cố định), Accuracy tăng dần từ $69.64\%$ lên $76.44\%$.
   - Các head phân hóa chức năng rõ rệt: có head tập trung vào từ lân cận, có head phân tán toàn câu, có head chú ý từ cảm xúc.
3. **Chi phí $O(T^2)$ của Attention:**
   - Khi $T$ tăng từ $32 \to 256$, kích thước ma trận attention tăng gấp 64 lần ($1,024 \to 65,536$), chứng minh rào cản tính toán trên sequence dài.
4. **Chiến lược Pooling:**
   - Khi huấn luyện từ đầu không có pretraining, **Masked Mean Pooling** ($74.44\%$) vượt trội hoàn toàn so với **First-token Pooling** ($53.04\%$).
5. **Cảnh báo phương pháp luận:**
   - Trọng số Attention chỉ thể hiện hành vi tính toán của mô hình (Model Behavior), không đồng nghĩa với lời giải thích nhân quả (Explanation) của con người.

---

## 5. Hướng dẫn Tái lập Kết quả (How to Run)

```bash
# 1. Kích hoạt môi trường conda
conda activate ml

# 2. Chạy module kiểm chứng Positional Encoding
python lab07/positional_encoding.py

# 3. Chạy kiểm tra kích thước Tensor của Transformer Block
python lab07/transformer_block.py

# 4. Chạy toàn bộ Notebook thực nghiệm
jupyter notebook lab07/experiments.ipynb
```

