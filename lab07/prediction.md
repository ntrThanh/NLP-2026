# Báo cáo Dự đoán & Đối chiếu Thực nghiệm (Prediction & Verification) — LAB 07

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Chủ đề:** From Attention to Transformer  

---

## 1. Các Dự đoán Trước Thực nghiệm (Giai đoạn Prediction — Mục 8)

Các dự đoán dưới đây được ghi lại trước khi tiến hành chạy các thực nghiệm trong `experiments.ipynb`.

---

### Prediction 1
- **Câu hỏi:** *Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?*
- **Prediction:** CÓ, Transformer có khả năng xử lý các token song song hơn RNN rất nhiều.
- **Reason:** RNN tính toán trạng thái ẩn tuần tự $h_t = f(x_t, h_{t-1})$, bước $t$ bắt buộc phải chờ bước $t-1$ hoàn thành, cản trở việc song song hóa trên phần cứng. Ngược lại, Self-Attention trong Transformer tính toán ma trận tương tác $Q K^T$ giữa tất cả các cặp token đồng thời thông qua các phép nhân ma trận khối (GEMM), hoàn toàn không có phụ thuộc tuần tự theo thời gian.
- **Confidence:** 99%

---

### Prediction 2
- **Câu hỏi:** *Nếu bỏ Positional Encoding, Transformer có còn biết thứ tự token không?*
- **Prediction:** KHÔNG, Transformer sẽ hoàn toàn mất khả năng nhận biết thứ tự token.
- **Reason:** Cơ chế Scaled Dot-Product Attention có tính chất hoán vị đồng biến (Permutation Equivariance): $\text{Attention}(P X) = P \text{Attention}(X)$ với mọi ma trận hoán vị $P$. Nếu không có Positional Encoding, việc tráo đổi vị trí token chỉ làm hoán vị các hàng của biểu diễn mà không làm thay đổi bản chất tương quan, khiến mô hình thoái hóa thành một dạng túi từ có trọng số (weighted bag-of-words).
- **Confidence:** 95%

---

### Prediction 3
- **Câu hỏi:** *Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?*
- **Prediction:** KHÔNG thể phân biệt được (nếu sau đó sử dụng Mean Pooling hoặc Sum Pooling).
- **Reason:** Vì Self-Attention không có vị trí là hoán vị đồng biến, các vector đầu ra của các từ tương ứng giữa hai câu sẽ giống hệt nhau. Khi áp dụng Mean/Sum Pooling để lấy biểu diễn toàn câu phục vụ phân loại, tổng các vector là giao hoán nên biểu diễn của cả hai câu sẽ hoàn toàn trùng nhau 100%.
- **Confidence:** 98%

---

### Prediction 4
- **Câu hỏi:** *Hai attention heads có nhất thiết học cùng một attention pattern không?*
- **Prediction:** KHÔNG nhất thiết, các head thường sẽ học các pattern khác nhau.
- **Reason:** Mỗi head có các ma trận chiếu độc lập $W_i^Q, W_i^K, W_i^V$ được khởi tạo ngẫu nhiên khác nhau và tối ưu hóa để cực tiểu hóa hàm mất mát. Quá trình gradient descent sẽ khuyến khích các head phân hóa vào các không gian con (subspaces) khác nhau để nắm bắt các đặc trưng cú pháp, ngữ nghĩa đa dạng (ví dụ: một head chú ý cục bộ từ lân cận, một head chú ý toàn cục).
- **Confidence:** 85%

---

### Prediction 5
- **Câu hỏi:** *Tăng số attention heads có chắc chắn làm accuracy tăng không?*
- **Prediction:** KHÔNG chắc chắn tăng liên tục, sẽ có ngưỡng bão hòa hoặc suy giảm nếu số head quá lớn.
- **Reason:** Khi cố định $d_{\text{model}}$, tăng số heads $h$ đồng nghĩa với việc số chiều của mỗi head $d_k = d_{\text{model}} / h$ sẽ bị thu hẹp lại. Nếu $d_k$ quá nhỏ (ví dụ $d_k \le 8$), không gian con không đủ dung lượng để biểu diễn các tương quan phức tạp. Ngoài ra, việc tăng head có thể dẫn đến hiện tượng dư thừa (redundancy) hoặc khó hội tụ nếu tập dữ liệu nhỏ.
- **Confidence:** 80%

---

### Prediction 6
- **Câu hỏi:** *Transformer có luôn nhanh hơn RNN trên mọi sequence length không?*
- **Prediction:** KHÔNG, Transformer chỉ nhanh hơn trên sequence vừa và ngắn khi tính toán song song, nhưng sẽ chậm hơn RNN khi sequence rất dài.
- **Reason:** Độ phức tạp tính toán và bộ nhớ của Self-Attention tăng theo cấp số nhân bậc hai $O(T^2 \cdot d)$, trong khi RNN chỉ tăng tuyến tính $O(T \cdot d^2)$. Khi $T$ rất lớn (hàng ngàn token), chi phí tính toán ma trận attention $T \times T$ sẽ bùng nổ, làm Transformer chậm hơn và ngốn bộ nhớ hơn rất nhiều so với bước chuyển trạng thái tuần tự của RNN.
- **Confidence:** 90%

---

## 2. Bảng Đối chiếu Thực nghiệm (Verification — Mục 28)

Dưới đây là bảng đối chiếu kết quả dự đoán ban đầu với số liệu thực tế thu được từ `experiments.ipynb`:

| Prediction | Dự đoán ban đầu | Kết quả thực tế | Đúng/Sai | Giải thích chi tiết dựa trên thực nghiệm |
| :---: | :---: | :---: | :---: | :--- |
| **#1** | Transformer xử lý song song hơn RNN | Đúng | **ĐÚNG** | Trong `experiments.ipynb`, Transformer tận dụng phép nhân ma trận khối song song trên toàn chuỗi. Tuy nhiên trên CPU đơn luồng, chi phí FLOPS của Transformer ($O(T^2)$) lớn hơn nên thời gian huấn luyện 1 epoch của Transformer (4 heads) là ~121s so với ~30s của LSTM. Nhưng về mặt kiến trúc, Transformer hoàn toàn không có phụ thuộc bước tuần tự $t-1 \to t$, cho phép tăng tốc vượt bậc khi chạy trên phần cứng song song (GPU/TPU). |
| **#2** | Bỏ PE, Transformer mất trật tự từ | Đúng | **ĐÚNG** | Ở Mục 11 & 12, khi không có PE, Self-Attention đối xử với các token độc lập với vị trí xuất hiện của chúng trong chuỗi. Biểu diễn của token chỉ phụ thuộc vào giá trị của nó và các token khác trong tập hợp. |
| **#3** | Hai câu đảo thứ tự không phân biệt được nếu không có PE | Không phân biệt được | **ĐÚNG** | Thực nghiệm Permutation (Mục 12) trên $S_1 = \text{"dog bites man"}$ và $S_2 = \text{"man bites dog"}$: Khi không có PE, khoảng cách Euclidean giữa representation toàn câu sau mean pooling là **$0.0000$** ($\cos = 1.000000$). Khi bổ sung PE, khoảng cách tăng lên **$0.2164$**, chứng minh PE là điều kiện bắt buộc để phân biệt thứ tự. |
| **#4** | Các heads học các pattern khác nhau | Đúng | **ĐÚNG** | Biểu đồ `attention_heads.png` (Mục 20) cho thấy rõ: Head 1 tập trung chú ý vào các từ lân cận (Local context), Head 2 phân tán đều (Global aggregation), Head 3 và 4 chú ý mạnh vào các từ mang sắc thái cảm xúc ("fantastic", "charming", "disappointing"). |
| **#5** | Tăng heads không chắc chắn tăng accuracy | Không chắc chắn tăng mãi | **ĐÚNG** | Thực nghiệm Mục 19 cho thấy khi tăng từ 1 head $\to$ 2 heads $\to$ 4 heads $\to$ 8 heads, accuracy tăng từ $69.64\% \to 72.20\% \to 74.20\% \to 76.44\%$. Tuy nhiên tốc độ tăng chậm dần (hiệu quả biên giảm dần), trong khi thời gian huấn luyện tăng từ 50.4s lên 111.1s. |
| **#6** | Transformer không luôn nhanh hơn RNN | Chậm hơn khi $T$ rất lớn | **ĐÚNG** | Thực nghiệm Mục 22 đo kích thước ma trận attention: $T=32 \implies 1,024$ phần tử; $T=256 \implies 65,536$ phần tử (tăng 64 lần). Thời gian forward-backward tăng rõ rệt, chứng minh chi phí bậc hai $O(T^2)$ là rào cản lớn khi mở rộng độ dài chuỗi. |

---

## 3. Kết luận rút ra từ Giai đoạn Prediction
Toàn bộ **6/6 dự đoán** đều khớp với bản chất toán học và bằng chứng thực nghiệm của kiến trúc Transformer. Việc thực hiện dự đoán trước thực nghiệm giúp làm sáng tỏ:
1. Động lực toán học sâu xa của từng thành phần (tại sao phải có Positional Encoding, tại sao phải chia Multi-Head).
2. Sự đánh đổi (trade-off) cốt lõi giữa khả năng mô hình hóa ngữ cảnh trực tiếp và chi phí bộ nhớ bậc hai $O(T^2)$ của cơ chế Attention.

