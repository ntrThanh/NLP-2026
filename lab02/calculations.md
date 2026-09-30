# Lab 02 — Theory & Calculation Exercises

> **Bản scan bài làm viết tay đầy đủ:** Chi tiết xem tại file [NguyenTrongThanh_calculations.pdf](file:///media/trong-thanh/Data2/University/Fourth%20Year/NLP/lab02/NguyenTrongThanh_calculations.pdf).

---

## 1. Bài tập tính toán trên Toy Corpus (Mục 7)

Xét tập ngữ liệu đồ chơi (Toy Corpus):
1. `the cat eats fish`
2. `the cat likes fish`
3. `the dog eats meat`

### Bài 1 — Unigram Language Model

**1. Xác định Vocabulary ($V$):**
$$V = \{\text{cat}, \text{dog}, \text{eats}, \text{fish}, \text{likes}, \text{meat}, \text{the}\}$$
Kích thước từ vựng: $|V| = 7$.

**2. Tổng số token ($N$):**
Mỗi câu có 4 từ $\implies N = 4 \times 3 = 12$ tokens.

**3. Tính xác suất $P(w)$ theo Maximum Likelihood Estimation (MLE):**
$$P(w) = \frac{C(w)}{N}$$
- $C(\text{the}) = 3 \implies P(\text{the}) = \frac{3}{12} = 0.25$
- $C(\text{cat}) = 2 \implies P(\text{cat}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
- $C(\text{fish}) = 2 \implies P(\text{fish}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
- $C(\text{dog}) = 1 \implies P(\text{dog}) = \frac{1}{12} \approx 0.0833$
- $C(\text{eats}) = 2 \implies P(\text{eats}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
- $C(\text{likes}) = 1 \implies P(\text{likes}) = \frac{1}{12} \approx 0.0833$
- $C(\text{meat}) = 1 \implies P(\text{meat}) = \frac{1}{12} \approx 0.0833$

**4. Kiểm tra tổng xác suất:**
$$\sum_{w \in V} P(w) = \frac{3 + 2 + 2 + 1 + 2 + 1 + 1}{12} = \frac{12}{12} = 1.0 \quad (\text{Thỏa mãn chuẩn hóa})$$

---

### Bài 2 — Bigram Language Model

Ước lượng MLE cho Bigram:
$$P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$$

**1. Tính các xác suất có điều kiện:**
- $P(\text{cat} \mid \text{the}) = \frac{C(\text{the cat})}{C(\text{the})} = \frac{2}{3} \approx 0.6667$
- $P(\text{dog} \mid \text{the}) = \frac{C(\text{the dog})}{C(\text{the})} = \frac{1}{3} \approx 0.3333$
- $P(\text{eats} \mid \text{cat}) = \frac{C(\text{cat eats})}{C(\text{cat})} = \frac{1}{2} = 0.5$
- $P(\text{likes} \mid \text{cat}) = \frac{C(\text{cat likes})}{C(\text{cat})} = \frac{1}{2} = 0.5$

**2. Trả lời câu hỏi:**
*Tại sao tổng xác suất của các từ đứng sau `the` phải bằng 1 nếu vocabulary và context được xử lý đầy đủ?*

> **Giải thích:**
> Theo định nghĩa xác suất có điều kiện, khi ngữ cảnh $h = \text{"the"}$ đã xảy ra:
> $$\sum_{w \in V} P(w \mid \text{the}) = \sum_{w \in V} \frac{C(\text{the}, w)}{C(\text{the})} = \frac{\sum_{w \in V} C(\text{the}, w)}{C(\text{the})} = \frac{C(\text{the})}{C(\text{the})} = 1$$
> Toàn bộ các từ xuất hiện sau `the` trong không gian từ vựng $V$ vét cạn mọi khả năng chuyển trạng thái tiếp theo của mô hình Markov bậc 1. Do đó, phân phối xác suất có điều kiện $P(\cdot \mid \text{the})$ bắt buộc phải chuẩn hóa về 1.

---

### Bài 3 — Xác suất câu

Xét câu: $S = \text{"the cat eats fish"}$

Phân rã theo mô hình Bigram:
$$P(S) = P(\text{the}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{eats} \mid \text{cat}) \cdot P(\text{fish} \mid \text{eats})$$

Các giá trị thành phần:
- $P(\text{the}) = \frac{3}{12} = \frac{1}{4}$
- $P(\text{cat} \mid \text{the}) = \frac{2}{3}$
- $P(\text{eats} \mid \text{cat}) = \frac{1}{2}$
- $C(\text{eats fish}) = 1, C(\text{eats}) = 2 \implies P(\text{fish} \mid \text{eats}) = \frac{1}{2}$

Suy ra:
$$P(S) = \frac{1}{4} \times \frac{2}{3} \times \frac{1}{2} \times \frac{1}{2} = \frac{2}{48} = \frac{1}{24} \approx 0.041667$$

**Trả lời câu hỏi:**
*Nếu thêm một từ vào câu, xác suất của cả câu có thể tăng không?*

> **Giải thích:**
> **Không thể tăng.** Với mỗi từ mới $w_{T+1}$ thêm vào cuối chuỗi, xác suất câu mới sẽ là:
> $$P(w_1, \dots, w_T, w_{T+1}) = P(w_1, \dots, w_T) \cdot P(w_{T+1} \mid w_T)$$
> Vì $0 \le P(w_{T+1} \mid w_T) \le 1$, phép nhân này luôn làm xác suất giữ nguyên hoặc giảm đi (nghiêm ngặt là giảm nếu xác suất $< 1$).
> **Ý nghĩa thực tế:** Xác suất chuỗi $P(S)$ giảm đơn điệu theo độ dài của câu, nên không thể dùng trực tiếp $P(S)$ để so sánh hai câu có độ dài khác nhau. Để so sánh công bằng, cần chuẩn hóa theo độ dài chuỗi bằng metric **Perplexity** ($PP(S) = P(S)^{-1/N}$).

---

### Bài 4 — Sentence Ranking

So sánh xác suất của hai câu:
- $S_1 = \text{"the cat eats fish"}$
- $S_2 = \text{"the dog eats fish"}$

**Dự đoán lý thuyết trước khi chạy code:**
- $P(S_1) = P(\text{the}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{eats} \mid \text{cat}) \cdot P(\text{fish} \mid \text{eats}) = \frac{1}{4} \cdot \frac{2}{3} \cdot \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{24}$
- $P(S_2) = P(\text{the}) \cdot P(\text{dog} \mid \text{the}) \cdot P(\text{eats} \mid \text{dog}) \cdot P(\text{fish} \mid \text{eats})$
  - $P(\text{dog} \mid \text{the}) = \frac{1}{3}$
  - Trong ngữ liệu: `the dog eats meat` $\implies C(\text{dog eats}) = 1, C(\text{dog}) = 1 \implies P(\text{eats} \mid \text{dog}) = 1.0$
  - $\implies P(S_2) = \frac{1}{4} \times \frac{1}{3} \times 1.0 \times \frac{1}{2} = \frac{1}{24} \approx 0.041667$

Mặc dù hai câu có cùng xác suất $1/24$ trên toy corpus siêu nhỏ, nhưng về mặt tự nhiên, cụm `the cat` xuất hiện phổ biến gấp đôi `the dog` ($P(\text{cat} \mid \text{the}) = 2/3 > P(\text{dog} \mid \text{the}) = 1/3$), trong khi `dog eats meat` làm cho transition $P(\text{eats} \mid \text{dog}) = 1.0$ bị overfit do dữ liệu về `dog` quá ít.

---

## 2. Bài tập suy luận trước khi Smoothing (Mục 9)

Cho corpus:
1. `I like NLP`
2. `I like AI`
3. `I study NLP`

**1. Count của bigram `study AI`:**
$$C(\text{study AI}) = 0$$

**2. Xác suất MLE:**
$$P_{\text{MLE}}(\text{AI} \mid \text{study}) = \frac{C(\text{study AI})}{C(\text{study})} = \frac{0}{1} = 0.0$$

**3. Điều gì xảy ra khi tính xác suất câu chứa bigram này?**
Vì xác suất câu là tích các xác suất có điều kiện:
$$P(S) = \prod_{t=1}^T P(w_t \mid w_{t-1})$$
Chỉ cần một thành phần bằng 0 thì toàn bộ $P(S) = 0$, và $\log P(S) = -\infty$, $PP(S) = +\infty$.

**4. Điều này có nghĩa mô hình “biết” rằng câu đó chắc chắn không thể xảy ra không?**
> **Trả lời:** Hoàn toàn **không**. 
> Việc $C(\text{study AI}) = 0$ chỉ là do hạn chế của kích thước mẫu thu thập (Sampling Sparsity). 
> Trong thực tế, *"I study AI"* là một câu hoàn toàn đúng ngữ pháp và mang ý nghĩa thực tế cao. 
> Hiện tượng này minh chứng cho nguyên lý quan trọng: **"Không quan sát thấy $\neq$ Không thể xảy ra"**. Đây là động lực bắt buộc phải sử dụng kỹ thuật **Smoothing** (Làm mịn xác suất).

---

## 3. Bài tập tính Smoothing — Add-one / Laplace (Mục 11)

Công thức Laplace smoothing:
$$P_{\text{Laplace}}(w \mid h) = \frac{C(h, w) + 1}{C(h) + V}$$
Cho context $h = \text{"cat"}$, $C(\text{cat}) = 10$, $|V| = 5$.

**1. Khi $C(\text{cat eats}) = 0$:**
$$P_{\text{Laplace}}(\text{eats} \mid \text{cat}) = \frac{0 + 1}{10 + 5} = \frac{1}{15} \approx 0.0667$$

**2. Khi $C(\text{cat eats}) = 3$:**
$$P_{\text{Laplace}}(\text{eats} \mid \text{cat}) = \frac{3 + 1}{10 + 5} = \frac{4}{15} \approx 0.2667$$
*(So với MLE: $P_{\text{MLE}} = \frac{3}{10} = 0.3000$)*.

**3. Smoothing đã thay đổi xác suất của những bigram khác như thế nào?**
> **Giải thích:**
> Smoothing hoạt động theo cơ chế chiết khấu xác suất (probability mass discounting). Nó "cắt bớt" một phần xác suất từ các bigram xuất hiện thường xuyên (ví dụ từ $0.3000$ giảm xuống $0.2667$) để phân phối lại cho các bigram chưa từng quan sát ($C = 0$, được gán $1/15$). Quá trình này làm phẳng phân phối (flattening) mà vẫn bảo toàn tổng xác suất bằng 1 trên toàn từ vựng $V$.

---

## 4. Bài tập tính Perplexity (Mục 18)

Công thức Perplexity cho chuỗi $W$ gồm $N$ token:
$$PP(W) = P(W)^{-\frac{1}{N}} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid \text{context}_i)\right)$$

Cho chuỗi 3 token ($N = 3$) với:
- $P(w_1) = 0.5$
- $P(w_2 \mid w_1) = 0.25$
- $P(w_3 \mid w_2) = 0.5$

**1. Tính $P(W)$ và $PP(W)$:**
$$P(W) = 0.5 \times 0.25 \times 0.5 = 0.0625 = \frac{1}{16}$$
$$PP(W) = \left(\frac{1}{16}\right)^{-\frac{1}{3}} = 16^{\frac{1}{3}} \approx 2.5198$$

**2. Tính lại khi $P(w_2 \mid w_1) = 0.1$:**
$$P(W) = 0.5 \times 0.1 \times 0.5 = 0.025 = \frac{1}{40}$$
$$PP(W) = \left(\frac{1}{40}\right)^{-\frac{1}{3}} = 40^{\frac{1}{3}} \approx 3.4199$$

**3. Vì sao chỉ một xác suất nhỏ cũng có thể làm Perplexity thay đổi đáng kể?**
> **Giải thích:**
> Perplexity tỷ lệ nghịch với căn bậc $N$ của tích các xác suất:
> $$PP(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid h_i) \right)$$
> Do hàm số $\ln(p) \to -\infty$ khi $p \to 0^+$, chỉ cần một xác suất thành phần nhỏ sẽ làm giá trị $-\ln(p)$ tăng vọt. Sau khi nhân với hàm số mũ $\exp(\cdot)$, mức phạt sẽ bùng nổ. Perplexity phản ánh độ "bối rối / bất ngờ" của mô hình, do đó một từ bất thường trong chuỗi sẽ ngay lập tức làm mô hình cực kỳ bối rối.
