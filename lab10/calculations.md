# Lab 10 — Theory & Calculation Exercises

## 0. Ba loại Prediction Head (Mục 5)

| Task | Input | Output | Head trên hidden states $H \in \mathbb{R}^{T \times d}$ | Shape logits |
|:--|:--|:--|:--|:--|
| Classification (Lab 08) | Text | 1 label | $\text{Linear}(h_{[CLS]})$ | $(B, C)$ |
| NER (Lab 09) | Text | 1 label / token | $\text{Linear}(h_t)$ cho mọi $t$ | $(B, T, C)$ |
| Extractive QA (Lab 10) | Question + Context | Start + End | $\text{Linear}_{d \to 2}(h_t)$ cho mọi $t$, tách thành 2 cột | $(B, T)$ + $(B, T)$ |

Với QA, head chỉ là một lớp `Linear(768, 2)`:

    logits = H @ W + b        # W: (768, 2), b: (2,)  →  logits: (B, T, 2)
    start_logits = logits[..., 0]   # (B, T)
    end_logits   = logits[..., 1]   # (B, T)
    P_start = softmax(start_logits, dim=T)     P_end = softmax(end_logits, dim=T)

Điểm khác biệt mấu chốt: ở NER, softmax chạy **theo chiều nhãn** (mỗi token chọn 1 trong C lớp); ở QA, softmax chạy **theo chiều vị trí** (cả chuỗi chọn ra 1 token làm start và 1 token làm end). "Số lớp" của QA chính là độ dài chuỗi $T$.

---

## Calculation Exercise 1 — Start / End

Context đã tokenize (chỉ tính riêng context, đánh số từ 0):

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Token | The | capital | of | France | is | Paris | . |

**(a) Answer = "Paris"**

$$start = 5, \quad end = 5$$

Answer là 1 token nên start = end (điều kiện $i \le j$ thỏa mãn với dấu bằng).

**(b) Answer = "capital of France"**

$$start = 1 \;(\text{capital}), \quad end = 3 \;(\text{France})$$

$$answer = context[1:3] = [\text{capital}, \text{of}, \text{France}]$$

Lưu ý: ở đây $end$ là **inclusive** (token cuối cùng thuộc answer). Nếu viết theo slice Python thì phải là `tokens[1:4]`.

**(c) Vị trí thực tế trong input của model.** Model không nhận riêng context mà nhận chuỗi ghép `[CLS] question [SEP] context [SEP]`. Với `bert-base-cased` (đã kiểm tra trên server22, xem `tokenizer_alignment.ipynb`):

| Idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Token | [CLS] | What | is | the | capital | of | France | ? | [SEP] | The | capital | of | France | is | Paris | . | [SEP] |
| sequence_id | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – | 1 | 1 | 1 | 1 | 1 | 1 | 1 | – |

Do đó nhãn huấn luyện thật sự là:
- "Paris": `start_positions = 14`, `end_positions = 14` (= 5 + 9, vì phần context bắt đầu ở index 9).
- "capital of France": `start_positions = 10`, `end_positions = 12`.

Bài học: start/end phải tính **trên chuỗi token đầu vào của model**, không phải trên context gốc. Offset của phần context = 1 ([CLS]) + số token question + 1 ([SEP]).

---

## Calculation Exercise 2 — Start Loss

Gold start = token 5. Phân phối dự đoán:

| Position | 0 | 1 | 2 | 3 | 4 | **5** | 6 | Tổng |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| $P_{start}$ | 0.02 | 0.05 | 0.03 | 0.10 | 0.15 | **0.60** | 0.05 | 1.00 ✓ |

$$L_{start} = -\log P_{start}(5) = -\ln(0.60) = 0.5108$$

(Dùng log tự nhiên như `torch.nn.CrossEntropyLoss`.) Chỉ xác suất tại vị trí đúng tham gia vào loss; các vị trí khác ảnh hưởng gián tiếp qua mẫu số của softmax.

**Tương tự với end.** Giả sử gold end cũng là token 5 ("Paris" 1 token) và model dự đoán:

| Position | 0 | 1 | 2 | 3 | 4 | **5** | 6 | Tổng |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| $P_{end}$ | 0.01 | 0.02 | 0.02 | 0.05 | 0.05 | **0.70** | 0.15 | 1.00 ✓ |

$$L_{end} = -\ln(0.70) = 0.3567$$

**So sánh một số tình huống** (tham chiếu để hiểu độ lớn loss):

| $P(\text{gold})$ | 1.00 | 0.90 | 0.60 | 0.50 | 0.15 | 1/7 (đều) | 0.02 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| $-\ln P$ | 0.000 | 0.105 | 0.511 | 0.693 | 1.897 | 1.946 | 3.912 |

Phân phối đều trên 7 vị trí cho loss $\ln 7 = 1.946$; với chuỗi thật dài $T = 384$ thì loss khởi điểm khoảng $\ln 384 \approx 5.95$ — đây là giá trị ta kỳ vọng thấy ở bước huấn luyện đầu tiên.

---

## QA Loss (Mục 8)

$$L = \frac{L_{start} + L_{end}}{2} = \frac{0.5108 + 0.3567}{2} = \frac{0.8675}{2} = 0.4338$$

Đây cũng chính là công thức mà `BertForQuestionAnswering` dùng (`total_loss = (start_loss + end_loss) / 2`); trong `qa_model.py` loss được tính tường minh bằng hai `CrossEntropyLoss` riêng để kiểm chứng.

### Vì sao cần hai prediction (start và end) thay vì một?

1. **Span là một cặp $(i, j)$, không phải một điểm.** Một phân phối duy nhất trên $T$ vị trí chỉ chọn được 1 token, nên chỉ biểu diễn được answer dài 1 token. Answer SQuAD trung bình ~3 từ và có thể dài >10 token.
2. **Cách thay thế "một prediction" là phân loại trực tiếp trên mọi span** $(i, j)$ với $i \le j$: số lớp là $T(T+1)/2$. Với $T = 384$ đó là **73,920** lớp cho mỗi mẫu, và phải tạo biểu diễn cho từng span → tốn bộ nhớ $O(T^2)$, dữ liệu huấn luyện cho mỗi lớp cực thưa.
3. **Phân rã start/end** giả định $P(i, j) \approx P_{start}(i) \cdot P_{end}(j)$, chỉ cần $2T$ logits ($O(T)$) từ một `Linear(d, 2)`. Mỗi đầu học một khái niệm có ý nghĩa ngôn ngữ riêng: "đâu là nơi câu trả lời bắt đầu" (thường sau giới từ, mạo từ, dấu phẩy…) và "đâu là nơi nó kết thúc" (trước dấu chấm, dấu phẩy, động từ…).
4. **Giải mã**: chọn $(i, j) = \arg\max_{i \le j,\; j - i < L_{max}} \big(s_i + e_j\big)$ (tổng logit = log-xác suất chưa chuẩn hóa). Ràng buộc $i \le j$ và độ dài tối đa được áp dụng lúc giải mã chứ không phải lúc huấn luyện.
5. Hai loss độc lập cũng cho phép **chẩn đoán riêng** lỗi wrong-start và wrong-end (Mục 25) — điều mà một phân loại span duy nhất không làm được.

---

## Calculation Exercise 3 — Character Span

Context: `The capital of France is Paris.` (31 ký tự, index 0 → 30)

```
idx : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
char: T h e _ c a p i t a l  _  o  f  _  F  r  a  n  c  e  _  i  s  _  P  a  r  i  s  .
```

1. **Index ký tự đầu tiên của "Paris"**: `answer_start = 25` (ký tự `P`).
2. **Index ký tự cuối cùng**: `29` (ký tự `s`).
3. **Character span**: inclusive là $[25, 29]$; theo quy ước exclusive-end của Python / offset mapping là $[25, 30)$:
   $$answer\_end = answer\_start + len(\text{"Paris"}) = 25 + 5 = 30$$
   Kiểm tra: `context[25:30] == "Paris"` ✓.
4. **Token tương ứng**: offset mapping của `bert-base-cased` cho phần context:

   | Token idx | 9 | 10 | 11 | 12 | 13 | **14** | 15 |
   |:--|:--|:--|:--|:--|:--|:--|:--|
   | Token | The | capital | of | France | is | **Paris** | . |
   | Offset | (0,3) | (4,11) | (12,14) | (15,21) | (22,24) | **(25,30)** | (30,31) |

   "Paris" nằm trọn trong một token, offset `(25, 30)` trùng khớp đúng với character span.
5. **Start token**: token có `start ≤ 25 < end` → token **14** (`25 ≤ 25 < 30`).
6. **End token**: token có `start < 30 ≤ end` → token **14** (`25 < 30 ≤ 30`).

Tóm tắt ba loại chỉ số cho cùng một đáp án "Paris":

| Loại chỉ số | Giá trị | Đơn vị |
|:--|:--|:--|
| Character index | 25 → 30 (exclusive) | ký tự trong chuỗi context |
| Word index (tách theo khoảng trắng) | 5 (`Paris.` — dính dấu chấm!) | từ trong context |
| Token index (trong input của model) | 14 → 14 | subword token sau `[CLS] question [SEP]` |

Ba chỉ số này khác nhau hoàn toàn: nếu dùng nhầm `answer_start = 25` làm token index thì nhãn sẽ trỏ ra ngoài cả chuỗi 17 token. Word index cũng không dùng được vì `"Paris."` dính dấu câu. Chỉ có **offset mapping** mới cầu nối chính xác ký tự ↔ token.

---

## Token-level F1 — tự tính (Mục 19)

Chuẩn hóa theo SQuAD: lowercase, bỏ dấu câu, bỏ mạo từ (`a`, `an`, `the`), gộp khoảng trắng.

**Ví dụ 1** — Gold: `Paris France`, Prediction: `France`

- Gold tokens = {paris, france} (2), Pred tokens = {france} (1), overlap = 1
- $P = 1/1 = 1.000$, $R = 1/2 = 0.500$
- $F1 = \dfrac{2 \cdot 1.0 \cdot 0.5}{1.0 + 0.5} = \dfrac{1.0}{1.5} = 0.667$, EM = 0

**Ví dụ 2** — Gold: `Paris`, Prediction: `the city of Paris`

- Sau chuẩn hóa: gold = [paris] (1), pred = [city, of, paris] (3) — "the" bị loại, overlap = 1
- $P = 1/3 = 0.333$, $R = 1/1 = 1.000$
- $F1 = \dfrac{2 \cdot 0.333 \cdot 1.0}{0.333 + 1.0} = \dfrac{0.667}{1.333} = 0.500$, EM = 0

**Ví dụ 3** — Gold: `New York City`, Prediction: `New York` (wrong end)

- gold = [new, york, city], pred = [new, york], overlap = 2
- $P = 2/2 = 1.0$, $R = 2/3 = 0.667$, $F1 = \dfrac{2 \cdot 1.0 \cdot 0.667}{1.667} = 0.800$, EM = 0

**Ví dụ 4** — Gold: `the Paris`, Prediction: `Paris.` → sau chuẩn hóa cả hai đều là `paris` → EM = 1, F1 = 1.0. Normalization giúp không phạt những khác biệt vô nghĩa (mạo từ, dấu câu, viết hoa).

Với nhiều gold answers (SQuAD val có tới 3 annotator), lấy **max** EM và **max** F1 trên các gold, rồi trung bình trên toàn tập. Kết quả tự tính được kiểm tra lại bằng hàm `squad_f1` trong `qa_model.py` (xem `tokenizer_alignment.ipynb`, mục cuối).
