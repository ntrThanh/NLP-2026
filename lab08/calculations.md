# Lab 08 — Theory & Calculation Exercises

**Học phần:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
**Học kỳ:** Học kỳ I - 2026  
**Chủ đề:** Pretrained Transformers and BERT Fine-Tuning  

---

## Mục 8 & 9. Calculation Exercise 1 — Masked Language Modeling (MLM) Loss

### 1. Đề bài và Thiết lập bài toán

Xét câu ngữ cảnh ban đầu:
$$\text{The cat eats fish.}$$

Sau khi thực hiện che (masking) tại vị trí động từ:
$$\text{The cat [MASK] fish.}$$

Mô hình BERT cần dự đoán phân phối xác suất có điều kiện trên toàn bộ từ vựng:
$$P(w_t \mid w_{\neq t})$$

Giả sử chỉ có một vị trí bị che ($|M| = 1$), và mô hình đưa ra phân phối xác suất dự đoán sau tầng Softmax tại vị trí `[MASK]` như sau:

| Token ($w$) | Xác suất dự đoán $P(w \mid x)$ |
| :--- | :--- |
| `cat` | $0.10$ |
| `dog` | $0.05$ |
| **`eats`** | **$0.70$** |
| `fish` | $0.15$ |

*Token nhãn thực tế (ground-truth):* $y = \text{eats}$.

---

### 2. Tính toán Loss ban đầu

Hàm mất mát Cross-Entropy tổng quát cho bài toán MLM được định nghĩa:
$$L = - \sum_{i \in M} \log P(y_i \mid x)$$

Do bài toán chỉ có một token bị mask ($M = \{t\}$) với nhãn đúng là $y_t = \text{eats}$:
$$L = - \log P(\text{eats} \mid x) = - \log(0.70)$$

#### Kết quả tính toán:

1. **Theo Logarit tự nhiên ($\ln$, cơ số $e$) — Chuẩn mực trong Machine Learning / PyTorch (`nn.CrossEntropyLoss`):**
   $$L = - \ln(0.70) \approx -(-0.356675) \approx \mathbf{0.3567}$$

2. **Theo Logarit cơ số 2 ($\log_2$) — Thường dùng trong Lý thuyết thông tin (đơn vị: bits / shannons):**
   $$L = - \log_2(0.70) \approx \mathbf{0.5146} \text{ bits}$$

3. **Theo Logarit thập phân ($\log_{10}$):**
   $$L = - \log_{10}(0.70) \approx \mathbf{0.1549}$$

---

### 3. Trả lời các câu hỏi phân tích

#### Câu hỏi 1: Nếu model dự đoán $P(\text{eats}) = 0.95$, loss thay đổi như thế nào?

- **Tính toán giá trị mới:**
  $$L_{\text{new}} = - \ln(0.95) \approx \mathbf{0.0513}$$
  *(Theo $\log_2$: $L_{\text{new}} = - \log_2(0.95) \approx 0.0740\text{ bits}$)*

- **Độ biến thiên của Loss:**
  $$\Delta L = L_{\text{new}} - L_{\text{old}} \approx 0.0513 - 0.3567 = -0.3054$$
  Loss **giảm mạnh** xấp xỉ **$85.6\%$** so với mức ban đầu.

- **Ý nghĩa & Bản chất:**
  - Xác suất dự đoán cho token đúng tăng từ $0.70$ lên $0.95$, phản ánh việc mô hình trở nên **tự tin và chính xác hơn rất nhiều**.
  - Do tính chất của hàm số $f(p) = -\ln(p)$ là hàm nghịch biến và tiệm cận $0$ khi $p \to 1$, khi xác suất của nhãn đúng tiến gần đến $1$, hàm phạt mất mát tiệm cận về $0$. 
  - Điều này đồng nghĩa gradient tính được sẽ rất nhỏ, báo hiệu mô hình đã học tốt mẫu dữ liệu này và không cần cập nhật tham số lớn.

---

#### Câu hỏi 2: Nếu model dự đoán $P(\text{eats}) = 0.01$, loss thay đổi như thế nào?

- **Tính toán giá trị mới:**
  $$L_{\text{new}} = - \ln(0.01) \approx \mathbf{4.6052}$$
  *(Theo $\log_2$: $L_{\text{new}} = - \log_2(0.01) \approx 6.6439\text{ bits}$)*

- **Độ biến thiên của Loss:**
  $$\Delta L = L_{\text{new}} - L_{\text{old}} \approx 4.6052 - 0.3567 = +4.2485$$
  Loss **tăng vọt gấp gần $13$ lần** ($12.91$ lần) so với giá trị ban đầu.

- **Ý nghĩa & Bản chất:**
  - Mô hình chỉ gán $1\%$ xác suất cho token đúng, đồng nghĩa mô hình đang gán tới $99\%$ xác suất vào các token sai khác (dự đoán sai với mức độ tự tin cao về nhãn sai).
  - Do tiệm cận đứng của hàm số:
    $$\lim_{p \to 0^+} (-\ln p) = +\infty$$
  - Khi xác suất của token đúng càng tiệm cận $0$, Cross-Entropy áp dụng một **hình phạt cực kỳ nặng (penalty rất lớn)**. Mức loss khổng lồ này sẽ sinh ra gradient lớn trong quá trình lan truyền ngược (backpropagation), buộc các trọng số của mô hình phải được điều chỉnh mạnh mẽ để khắc phục lỗi sai nghiêm trọng này.

---

#### Câu hỏi 3: Vì sao cross-entropy phù hợp với bài toán MLM?

Hàm mất mát Cross-Entropy là sự lựa chọn tiêu chuẩn và phù hợp nhất cho bài toán Masked Language Modeling vì các lý do sau:

1. **Bản chất bài toán MLM là bài toán phân loại đa lớp (Multi-class Classification) trên toàn từ vựng ($|V|$ lớp):**
   - Tại mỗi vị trí `[MASK]`, mô hình không dự đoán một giá trị liên tục mà phải chọn ra một token đúng duy nhất từ không gian từ vựng rời rạc gồm hàng chục nghìn token (ví dụ: từ điển BERT có $|V| = 30.522$ tokens).
   - Phân phối dự đoán của mô hình là một phân phối xác suất rời rạc chuẩn hóa qua hàm Softmax ($\sum_{w \in V} P(w) = 1$). Cross-Entropy là thước đo khoảng cách phân phối tự nhiên nhất giữa phân phối một điểm (one-hot vector của nhãn thật) và phân phối dự đoán của mạng nơ-ron.

2. **Tương đương với nguyên lý Cực đại hóa Hợp lý (Maximum Likelihood Estimation - MLE):**
   - Tối thiểu hóa hàm Cross-Entropy tương đương trực tiếp với việc tối đa hóa xác suất log (Log-Likelihood) của chuỗi văn bản tự nhiên theo ngữ cảnh:
     $$\max_{\theta} \sum_{t \in M} \log P_\theta(w_t \mid w_{\neq t})$$
   - Đây là nền tảng thống kê chuẩn mực để huấn luyện các mô hình ngôn ngữ (Language Models) có khả năng sinh và hiểu biểu diễn ngôn ngữ.

3. **Cơ chế phạt bất đối xứng dựa trên Lý thuyết Thông tin (Information Theory & Surprisal):**
   - Giá trị $-\log P(y \mid x)$ chính là **lượng tin bất ngờ (surprisal hay self-information)** của sự kiện $y$.
   - Nếu mô hình đã đoán đúng với xác suất cao ($P \approx 1$), lượng tin bất ngờ bằng 0 $\to$ không phạt. Nhưng nếu một sự kiện hiển nhiên đúng lại bị mô hình đánh giá xác suất gần 0, lượng tin bất ngờ tiến tới vô cùng $\to$ phạt cực đại. Cơ chế này khuyến khích mô hình tập trung phân biệt và sửa chữa những lỗi sai nghiêm trọng.

4. **Tính chất Gradient tối ưu khi kết hợp Softmax (Tránh bão hòa đạo hàm):**
   - Khi kết hợp hàm Softmax ở đầu ra với hàm Cross-Entropy Loss, đạo hàm của loss $L$ đối với logit đầu vào $z_i$ trước Softmax có dạng toán học cực kỳ đẹp và trực quan:
     $$\frac{\partial L}{\partial z_i} = P(w_i \mid x) - y_i$$
     *(trong đó $y_i = 1$ nếu $w_i$ là token đúng, và $y_i = 0$ nếu là token khác).*
   - Đạo hàm này không bị ảnh hưởng bởi hiện tượng bão hòa đạo hàm (vanishing gradient) như khi dùng MSE (Mean Squared Error) với hàm kích hoạt phi tuyến, giúp thuật toán Gradient Descent (AdamW) hội tụ nhanh chóng và ổn định qua hàng triệu bước pre-training.

---

## Tóm tắt bảng so sánh độ nhạy của Loss

| Kịch bản | Xác suất dự đoán $P(\text{eats})$ | Loss ($-\ln p$) | So với mức cơ sở ($p=0.70$) | Trạng thái mô hình |
| :--- | :--- | :--- | :--- | :--- |
| **Dự đoán kém (sai nghiêm trọng)** | $0.01$ | $\mathbf{4.6052}$ | Tăng $+4.2485$ (gấp $\approx 13$ lần) | Bị phạt rất nặng, gradient cực lớn để ép sửa sai |
| **Mức cơ sở (đề bài cho)** | $0.70$ | $\mathbf{0.3567}$ | $0.0000$ (Chuẩn) | Đã định hướng đúng token mục tiêu |
| **Dự đoán rất tốt (tự tin cao)** | $0.95$ | $\mathbf{0.0513}$ | Giảm $-0.3054$ (giảm $\approx 85.6\%$) | Mất mát tiệm cận 0, hội tụ ổn định |


---

## Mục 10. Calculation Exercise 2 — Tokenization

### 1. Đề bài

Cho chuỗi văn bản đầu vào:
$$\text{"The cats are playing."}$$

Thực hiện các yêu cầu:
1. Tokenize bằng tokenizer được cung cấp (`bert-base-uncased` — mô hình BERT chuẩn).
2. Ghi chuỗi token (`token sequence`).
3. Ghi danh sách token ID (`token IDs`).
4. Xác định các token đặc biệt (`special tokens`).
5. Xác định mặt nạ chú ý (`attention mask`).
6. So sánh số lượng: `words`, `tokens`, `subword tokens`.

---

### 2. Thực thi Tokenization với Pretrained Tokenizer

Đoạn mã Python thực thi với thư viện `transformers`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "The cats are playing."

# 1. Tokenize cơ bản (subwords)
raw_tokens = tokenizer.tokenize(text)

# 2. Tokenize đầy đủ có special tokens (và minh họa padding đến max_length = 10)
encoded = tokenizer(text)
tokens_with_special = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

# Minh họa kèm padding
encoded_padded = tokenizer(text, padding="max_length", max_length=10)
padded_tokens = tokenizer.convert_ids_to_tokens(encoded_padded["input_ids"])
```

---

### 3. Kết quả chi tiết từng câu hỏi

#### 1 & 2. Token Sequence

* **Chuỗi token nội dung (không tính special tokens):**
  $$\text{['the', 'cats', 'are', 'playing', '.']}$$
  *(Lưu ý: Do sử dụng `bert-base-uncased`, chữ viết hoa `"The"` được chuẩn hóa về chữ viết thường `"the"`).*

* **Chuỗi token hoàn chỉnh đưa vào BERT (kèm special tokens):**
  $$\text{['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]']}$$

* *(Mở rộng)* **Chuỗi token khi có padding (ví dụ `max_length = 10`):**
  $$\text{['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]']}$$

---

#### 3. Token IDs (`input_ids`)

Bảng tra cứu chi tiết từng token tương ứng với chỉ số trong từ điển WordPiece ($|V| = 30.522$):

| Index | Token | Token ID | Loại Token | Ghi chú |
| :---: | :---: | :---: | :---: | :--- |
| 0 | `[CLS]` | **101** | Special Token | Đánh dấu bắt đầu sequence |
| 1 | `the` | **1996** | Subword / Word | Chuyển từ "The" về viết thường |
| 2 | `cats` | **8870** | Subword / Word | Có sẵn trong vocab |
| 3 | `are` | **2024** | Subword / Word | Động từ to-be |
| 4 | `playing` | **2652** | Subword / Word | Động từ đuôi -ing |
| 5 | `.` | **1012** | Punctuation | Dấu chấm câu |
| 6 | `[SEP]` | **102** | Special Token | Đánh dấu kết thúc sequence |

* **Danh sách Token IDs (`input_ids` tiêu chuẩn):**
  $$[101, 1996, 8870, 2024, 2652, 1012, 102]$$

* *(Mở rộng khi đệm `max_length = 10`):*
  $$[101, 1996, 8870, 2024, 2652, 1012, 102, 0, 0, 0]$$

---

#### 4. Xác định Special Tokens

Trong chuỗi trên, các token đặc biệt bao gồm:

1. **`[CLS]` (Classification Token):**
   * *ID:* `101` (ở vị trí đầu tiên, index 0).
   * *Vai trò:* Là token đặc biệt bắt buộc đứng đầu mọi chuỗi đầu vào của BERT. Vector biểu diễn ẩn cuối cùng của `[CLS]` ở tầng trên cùng được dùng làm vector đại diện cho toàn bộ câu/văn bản để đưa vào Linear Classification Head (phân loại cảm xúc, phân loại chủ đề...).
2. **`[SEP]` (Separator Token):**
   * *ID:* `102` (ở vị trí cuối câu, index 6).
   * *Vai trò:* Đánh dấu ranh giới kết thúc của một câu hoặc phân tách hai câu độc lập trong tác vụ phân loại cặp câu (ví dụ: Sentence A và Sentence B trong NLI/QA).
3. **`[PAD]` (Padding Token — xuất hiện khi xử lý theo batch):**
   * *ID:* `0`.
   * *Vai trò:* Token đệm vào đuôi các chuỗi ngắn để mọi chuỗi trong cùng một batch có cùng kích thước tensor cố định.

---

#### 5. Xác định Attention Mask

Mặt nạ chú ý `attention_mask` là vector nhị phân dùng để chỉ thị cho cơ chế Self-Attention biết vị trí nào là token thật cần tính toán chú ý ($1$) và vị trí nào là padding cần bỏ qua ($0$).

* **Trường hợp không padding (độ dài thực tế $L = 7$):**
  $$\text{attention\_mask} = [1, 1, 1, 1, 1, 1, 1]$$
  *(Tất cả 7 vị trí đều hợp lệ, tham gia đầy đủ vào cơ chế Self-Attention).*

* **Trường hợp có padding (ví dụ với `max_length = 10`):**
  $$\text{attention\_mask} = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]$$
  * Các vị trí từ index 0 đến 6 có giá trị **$1$**: Mô hình tính toán Self-Attention bình thường.
  * Các vị trí từ index 7 đến 9 có giá trị **$0$**: Trong ma trận Attention Score trước khi qua Softmax, các vị trí này sẽ bị gán giá trị $-\infty$ (hoặc $-10000.0$), do đó xác suất attention gán cho các vị trí này sẽ bằng $0$ ($e^{-\infty} = 0$). Điều này đảm bảo nội dung của token padding không làm sai lệch biểu diễn ngữ cảnh của câu.

---

#### 6. So sánh số lượng: Words vs. Tokens vs. Subword Tokens

| Tiêu chí | Số lượng | Danh sách thành phần | Giải thích bản chất |
| :--- | :---: | :--- | :--- |
| **Words (Từ ngữ nghĩa)** | **4 từ** | `The`, `cats`, `are`, `playing` *(và 1 dấu câu `.`)* | Đếm theo đơn vị từ tự nhiên trong ngôn ngữ học (tách theo khoảng trắng / từ vựng từ điển). |
| **Tokens (Đầu vào mô hình BERT)** | **7 tokens** | `[CLS]`, `the`, `cats`, `are`, `playing`, `.`, `[SEP]` | Toàn bộ các đơn vị token rời rạc được đưa vào mạng nơ-ron, bao gồm cả 2 special tokens (`[CLS]` và `[SEP]`). |
| **Subword Tokens (Đơn vị từ con)** | **5 đơn vị** | `the`, `cats`, `are`, `playing`, `.` | Các mảnh đơn vị từ vựng sinh ra từ thuật toán WordPiece. |

#### Phân tích sâu về thuật toán Subword (WordPiece):
* Trong câu này, các từ `the`, `cats`, `are`, `playing` và dấu `.` đều là những từ rất phổ biến trong ngữ liệu tiếng Anh, nên chúng đã xuất hiện sẵn dưới dạng một mục từ độc lập (whole-word level) trong từ điển BERT ($30.522$ tokens) mà không bị bẻ vụn ra thành các subwords nhỏ hơn có tiền tố `##`.
* **Ví dụ đối chiếu nếu gặp từ hiếm (OOV):**
  * Nếu câu chứa từ hiếm như `"The catss are playingg."`, WordPiece sẽ phân rã:
    * `catss` $\to$ `['cats', '##s']`
    * `playingg` $\to$ `['playing', '##g']`
  * Khi đó số lượng subword tokens sẽ tăng lên thành $7$ (chưa kể special tokens), giúp mô hình không bao giờ gặp lỗi Unknown (`[UNK]`) mà vẫn bảo toàn được gốc từ ngữ nghĩa.
