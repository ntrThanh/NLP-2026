# Báo cáo Chiêm nghiệm & Trả lời Vấn đáp (Reflection & Learning Check) — LAB 07

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** From Attention to Transformer  

---

## 1. Mạch nối Tiến hóa Xuyên suốt: LAB 01 $\to$ LAB 07

Chương trình môn học được thiết kế với một lộ trình tiến hóa tự nhiên và chặt chẽ qua từng lab:

```
LAB 01 (TF-IDF): Biểu diễn từ điển thưa (Sparse Lexical) ──> Tìm kiếm thông tin
       │  (Hạn chế: Chiều lớn, thưa thớt, không nắm bắt ngữ nghĩa từ đồng nghĩa)
       ▼
LAB 02 (N-gram Language Model): Xác suất chuỗi cục bộ ──> Perplexity
       │  (Hạn chế: Lời nguyền số chiều, không nắm bắt ngữ cảnh vượt quá n từ)
       ▼
LAB 03 (Word2Vec): Biểu diễn dày (Dense Semantic Embeddings) ──> Tương đồng ngữ nghĩa
       │  (Hạn chế: Vector tĩnh, mỗi từ chỉ có 1 vector bất kể ngữ cảnh)
       ▼
LAB 04 (Text Classification): Bộ phân loại tĩnh (Bag-of-Words / Avg Word2Vec) ──> Baseline
       │  (Hạn chế: Mất hoàn toàn trật tự từ, không nhận diện được phạm vi phủ định)
       ▼
LAB 05 (RNN / LSTM / GRU): Mô hình tuần tự theo thời gian ──> Trạng thái ẩn h_t
       │  (Hạn chế: Phụ thuộc tuần tự t-1 -> t, không song song hóa, nghẽn thông tin)
       ▼
LAB 06 (Attention): Scaled Dot-Product & Self-Attention ──> Vector ngữ cảnh động Z
       │  (Hạn chế: Chưa có cấu trúc khối hoàn chỉnh, chưa có thông tin vị trí và FFN)
       ▼
LAB 07 (Transformer): Self-Attention + Positional Encoding + FFN + Residual/Norm
       │  (Hạn chế còn lại: Huấn luyện từ đầu cần nhiều dữ liệu; chi phí O(T^2))
       ▼
LAB 08 (Pretrained Transformer / BERT): Tiền huấn luyện quy mô lớn ──> Fine-tuning
```

---

### Bảng So sánh Tổng hợp Toàn bộ Các Phương pháp (Mục 27 W7.pdf)

| Thuộc tính (Property) | TF-IDF (LAB 01/04) | Word2Vec Avg (LAB 03/04) | LSTM (LAB 05) | Self-Attention (LAB 06) | Transformer (LAB 07) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Biểu diễn thưa (Sparse)** | Có ($\checkmark$) | Không ($\times$) | Không ($\times$) | Không ($\times$) | Không ($\times$) |
| **Biểu diễn dày (Dense)** | Không ($\times$) | Có ($\checkmark$) | Có ($\checkmark$) | Có ($\checkmark$) | Có ($\checkmark$) |
| **Bảo toàn trật tự từ (Word order)** | Không ($\times$) | Không ($\times$) | Có ($\checkmark$) | Giới hạn ($\times$) | Có ($\checkmark^*$) |
| **Tính phụ thuộc ngữ cảnh (Context-dependent)** | Không ($\times$) | Không ($\times$) | Có ($\checkmark$) | Có ($\checkmark$) | Có ($\checkmark$) |
| **Tương tác trực tiếp từng cặp (Direct interaction)**| Không ($\times$) | Không ($\times$) | Giới hạn ($\times$) | Có ($\checkmark$) | Có ($\checkmark$) |
| **Tính toán tuần tự (Sequential computation)** | Không ($\times$) | Không ($\times$) | Có ($\checkmark$) | Không ($\times$) | Không ($\times$) |
| **Khả năng tính toán song song (Parallelizable)** | Có ($\checkmark$) | Có ($\checkmark$) | Không ($\times$) | Có ($\checkmark$) | Có ($\checkmark$) |
| **Nắm bắt tương tác tầm xa (Long-range)** | Kém | Kém | Khó khăn | Tốt hơn | Rất tốt |
| **Cơ chế chú ý (Attention mechanism)** | Không ($\times$) | Không ($\times$) | Không ($\times$) | Có ($\checkmark$) | Có ($\checkmark$) |
| **Cần thông tin vị trí (Positional information)**| Không cần | Không cần | Tự thân có sẵn | Thiếu | Bắt buộc ($PE$) |
| **Độ phức tạp theo độ dài chuỗi ($T$)** | $O(T)$ | $O(T)$ | Tuyến tính $O(T \cdot d^2)$ | Bậc hai $O(T^2 \cdot d)$ | Bậc hai $O(T^2 \cdot d)$ |

*\* Transformer cần Positional Encoding ($PE$) để biểu diễn trật tự từ.*

---

## 2. Trả lời 9 Câu hỏi Vấn đáp Nhanh (Individual Learning Check — Mục 31)

### Q1: Tại sao Transformer cần Positional Encoding?
**Trả lời:** Bản thân cơ chế Self-Attention có tính chất **Permutation Equivariance** (hoán vị đồng biến): nó chỉ tính tích vô hướng nội suy giữa các vector token mà không hề có khái niệm về vị trí đứng trước hay đứng sau. Nếu hai câu có cùng tập từ vựng nhưng khác thứ tự (ví dụ: *"dog bites man"* và *"man bites dog"*), Self-Attention thuần túy sẽ tạo ra cùng các vector tương ứng và sau phép cộng/trung bình (pooling), biểu diễn của hai câu là hoàn toàn trùng khớp 100%. Positional Encoding ($PE$) cộng trực tiếp vào embedding của mỗi token một vector tần số duy nhất, phá vỡ tính đối xứng hoán vị và cung cấp thông tin về vị trí tuyệt đối cũng như tương đối của các từ.

---

### Q2: Multi-Head Attention khác một Attention Head ở đâu?
**Trả lời:** 
- Một attention head duy nhất chỉ có một bộ ma trận chiếu $(W^Q, W^K, W^V)$ và chỉ có thể học một kiểu tương quan ngữ cảnh cụ thể (ví dụ: chú ý vào từ lân cận).
- Multi-Head Attention thực hiện $h$ phép chiếu độc lập song song vào các không gian con có số chiều nhỏ hơn ($d_k = d_{\text{model}} / h$). Điều này cho phép mô hình đồng thời quan sát và kết hợp nhiều dạng quan hệ ngôn ngữ đa dạng tại cùng một thời điểm (ví dụ: head 1 học cú pháp chủ vị, head 2 học phạm vi phủ định, head 3 học liên kết liên từ). Sau đó, các head được nối ghép (`Concat`) và chiếu qua $W^O$ để tổng hợp thông tin liên-head.

---

### Q3: Nếu $d_{\text{model}} = 128$ và có $8$ heads thì $d_k$ bằng bao nhiêu?
**Trả lời:** 
$$d_k = \frac{d_{\text{model}}}{h} = \frac{128}{8} = 16$$
Mỗi head sẽ tính toán attention trên các vector Query, Key, Value có số chiều là $16$. Sau khi concat 8 heads lại, ta thu được vector có chiều $8 \times 16 = 128$, bằng đúng $d_{\text{model}}$ ban đầu.

---

### Q4: Tại sao ma trận Attention có kích thước $T \times T$?
**Trả lời:** Trong Self-Attention, mỗi token trong số $T$ tokens của chuỗi đóng vai trò là một **Query** và đồng thời cũng là một **Key**. Tích vô hướng $Q K^T$ là phép nhân giữa ma trận $Q \in \mathbb{R}^{T \times d_k}$ và ma trận $K^T \in \mathbb{R}^{d_k \times T}$, tạo ra ma trận kích thước $T \times T$. Phần tử ở hàng $i$, cột $j$ đại diện cho mức độ chú ý (attention weight) mà token thứ $i$ dành cho token thứ $j$.

---

### Q5: Residual Connection có vai trò gì trong Transformer Block?
**Trả lời:** Residual Connection ($y = x + \text{Sublayer}(x)$) có hai vai trò cốt lõi:
1. **Duy trì dòng chảy gradient (Gradient Flow):** Cung cấp một đường truyền trực tiếp (highway) giúp gradient lan truyền ngược về các tầng trước mà không bị tiêu biến (vanishing gradient), cho phép huấn luyện ổn định các mạng nơ-ron sâu hàng chục tới hàng trăm tầng.
2. **Bảo toàn thông tin nguyên bản:** Đảm bảo rằng thông tin của đầu vào $x$ không bị mất mát hay biến dạng hoàn toàn sau các phép biến đổi phi tuyến tính của Attention hoặc FFN; mô hình chỉ cần học phần thông tin điều chỉnh thặng dư (residual).

---

### Q6: Feed-Forward Network (FFN) làm gì trong Transformer Block?
**Trả lời:** Self-Attention chủ yếu thực hiện việc phân phối và pha trộn thông tin giữa các token khác nhau (token-to-token interaction) thông qua các tổ hợp tuyến tính có trọng số. Sau bước này, FFN áp dụng một mạng truyền thẳng 2 tầng (Linear $\to$ ReLU $\to$ Linear) tác động độc lập trên từng vị trí (position-wise). FFN chịu trách nhiệm thực hiện các phép biến đổi phi tuyến tính sâu để tinh chỉnh, nâng cấp và học các biểu diễn đặc trưng phức tạp của từng token.

---

### Q7: Tại sao Transformer có thể xử lý các token song song hơn RNN?
**Trả lời:** Trong RNN, trạng thái ẩn $h_t$ phụ thuộc đệ quy vào $h_{t-1}$ ($h_t = f(x_t, h_{t-1})$), do đó bước $t$ bắt buộc phải chờ bước $t-1$ tính xong (sequential bottleneck). Trong Transformer, tất cả $T$ vector Query, Key, Value của toàn chuỗi được tính cùng lúc qua một phép nhân ma trận $X W$, và ma trận attention $T \times T$ được tính đồng thời qua phép nhân khối $Q K^T$ bằng các phép toán GEMM được tối ưu hóa cực đại trên GPU/TPU, hoàn toàn không có phụ thuộc tuần tự theo thời gian.

---

### Q8: Nếu sequence dài gấp đôi, ma trận Attention thay đổi như thế nào?
**Trả lời:** Nếu độ dài sequence tăng từ $T$ lên $2T$, số phần tử trong ma trận attention sẽ tăng theo cấp số nhân bậc hai:
$$(2T)^2 = 4 T^2$$
Tức là số phần tử và chi phí bộ nhớ lưu trữ ma trận attention sẽ **tăng gấp 4 lần**. Đây chính là giới hạn bộ nhớ bậc hai $O(T^2)$ khiến các mô hình Transformer tiêu chuẩn khó mở rộng trực tiếp lên các văn bản siêu dài (hàng chục nghìn tokens).

---

### Q9: Vì sao không thể kết luận rằng attention weight cao đồng nghĩa token đó là “lý do” model đưa ra prediction?
**Trả lời:** Vì ba lý do khoa học:
1. **Attention chỉ là một bước trung gian:** Đầu ra của Attention ($Z = A V$) còn phải đi qua Residual Connection, LayerNorm, tầng Feed-Forward phi tuyến (ReLU), Pooling và Linear Classifier trước khi đưa ra nhãn dự đoán cuối cùng. Một đặc trưng có attention cao có thể bị triệt tiêu bởi tầng FFN hoặc hàm kích hoạt phía sau.
2. **Tương quan $\neq$ Nhân quả:** Attention weights chỉ thể hiện cách mô hình phân bổ trọng số tính toán nội bộ (model behavior), không phải là bằng chứng giải thích nhân quả logic theo tư duy con người (explanation).
3. **Hiện tượng nhiều bộ weights cho cùng một output:** Nhiều nghiên cứu thực nghiệm (như Jain & Wallace, 2019) đã chỉ ra rằng có thể tìm thấy các phân phối attention weights hoàn toàn khác nhau nhưng vẫn tạo ra cùng một kết quả dự đoán phân loại.

---

## 3. Tuyên bố Trợ giúp của AI (AI Assistance Statement — Mục 30)

Tuân thủ nghiêm ngặt chính sách AI của môn học:
- **Công cụ hỗ trợ (Tool):** Antigravity / Gemini Coding Assistant.
- **Mục đích sử dụng (Purpose):** Viết boilerplate training loop trong PyTorch, định dạng biểu đồ trực quan hóa matplotlib, hỗ trợ debugging tensor shapes.
- **Nội dung do AI hỗ trợ sinh (Generated):** Mã nguồn khung `experiments.ipynb`, script kiểm thử shape `transformer_block.py`.
- **Nội dung sinh viên tự thực hiện và kiểm chứng (Modified & Verified):** Toàn bộ bài tập tính toán trên giấy, kiểm chứng công thức toán học ma trận của Attention và FFN, phân tích các mẫu lỗi định tính trong `error_analysis.md`, trả lời các câu hỏi chiêm nghiệm trong `reflection.md` và kiểm tra tính toán tensor shape.

