# Tổng hợp kiến thức NLP — Lab 01 → Lab 05

> Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng (VNU-HUS, HK I - 2026).
> Tài liệu ôn tập tổng hợp từ đề bài W1–W5, bài làm Lab 01–04 (README, calculations, prediction, error analysis, reflection) và kiến thức nền liên quan.
> Lab 05 hiện mới có đề, nên phần đó tóm tắt lý thuyết và các bài tính trong đề, chưa có số liệu thực nghiệm.

---

## Mục lục

0. [Bức tranh tổng thể](#0-bức-tranh-tổng-thể)
1. [Lab 01 — Tiền xử lý, TF-IDF và tìm kiếm](#1-lab-01--tiền-xử-lý-tf-idf-và-tìm-kiếm)
2. [Lab 02 — Mô hình ngôn ngữ N-gram, Smoothing, Perplexity](#2-lab-02--mô-hình-ngôn-ngữ-n-gram-smoothing-perplexity)
3. [Lab 03 — Word Embeddings](#3-lab-03--word-embeddings)
4. [Lab 04 — Phân loại văn bản](#4-lab-04--phân-loại-văn-bản)
5. [Lab 05 — Sequence Models: RNN, LSTM, GRU](#5-lab-05--sequence-models-rnn-lstm-gru)
6. [Bảng so sánh xuyên suốt](#6-bảng-so-sánh-xuyên-suốt)
7. [Phương pháp làm lab (quy trình khoa học)](#7-phương-pháp-làm-lab-quy-trình-khoa-học)
8. [Ngân hàng câu hỏi vấn đáp](#8-ngân-hàng-câu-hỏi-vấn-đáp)
9. [Bảng công thức tra nhanh](#9-bảng-công-thức-tra-nhanh)

---

## 0. Bức tranh tổng thể

Mỗi lab giải quyết một giới hạn mà lab trước để lộ ra:

```text
LAB 01  Text → Tokenize → TF-IDF (sparse)      → Cosine → Search
        Giới hạn: không hiểu đồng nghĩa ("heart attack" vs "myocardial infarction": cos = 0)
   ↓
LAB 02  Text → N-gram counts → P(w | context)   → Language Model → Perplexity
        Giới hạn: data sparsity, chỉ nhớ n-1 từ, không biết đồng nghĩa
   ↓
LAB 03  Text → Context window → Dense embedding (Word2Vec) → Semantic similarity
        Giới hạn: mỗi từ chỉ có 1 vector (polysemy: "bank"), không có thứ tự
   ↓
LAB 04  Text → Representation (TF-IDF / Avg W2V) → Classifier → Evaluation → Error analysis
        Giới hạn: Bag-of-Words mất thứ tự từ ("not good" ≈ "good")
   ↓
LAB 05  Text → Token IDs → Embedding → RNN/LSTM/GRU (hidden state) → Classifier
        Giới hạn: thông tin phải đi qua chuỗi dài, vanishing gradient, không song song hoá
   ↓
LAB 06–07 (sắp tới)  Attention (Query/Key/Value) → Self-Attention → Transformer
```

Ba trục để so sánh mọi biểu diễn:

| Trục | Câu hỏi |
| :--- | :--- |
| Sparse ↔ Dense | Vector có nhiều số 0 (chiều = \|V\|) hay ít chiều, đa số khác 0? |
| Lexical ↔ Semantic | Hai từ đồng nghĩa có gần nhau không? |
| Static ↔ Contextual / Sequential | Biểu diễn có phụ thuộc thứ tự và ngữ cảnh xung quanh không? |

---

## 1. Lab 01 — Tiền xử lý, TF-IDF và tìm kiếm

**Câu hỏi trung tâm:** Một quyết định trong tiền xử lý ảnh hưởng thế nào đến biểu diễn văn bản và chất lượng tìm kiếm?

### 1.1. Pipeline

```text
Raw text → Tokenizer → CountVectorizer → TF → IDF → TF-IDF matrix → Cosine similarity → Ranking
```

### 1.2. Tiền xử lý (Preprocessing)

| Bước | Tác dụng | Rủi ro |
| :--- | :--- | :--- |
| Lowercasing | Gộp `Apple`/`apple`, giảm \|V\| (thực nghiệm: 240.792 → 186.582 từ) | Mất phân biệt tên riêng (`Apple` công ty vs quả táo) |
| Bỏ dấu câu / ký tự đặc biệt (regex) | Giảm nhiễu | Tách sai `cat's` → `cat` + `s`, mất `C++`, `3.14` |
| Bỏ stopwords | Bỏ từ chức năng (`the`, `of`…). Thực nghiệm: chỉ giảm ~0,5% vocabulary nhưng giảm ~1/3 tổng số token | Mất phủ định (`not`, `no`) — rất nguy hiểm cho sentiment (Lab 04) |
| Stemming / Lemmatization | Gộp `learned`/`learning` | Stemming có thể cắt sai (`university` → `univers`) |
| Subword tokenizer (WordPiece/BPE) | Vocabulary cố định (BERT: 30.522), **OOV = 0%** vì từ lạ bị tách thành mảnh | Một từ thành nhiều token, khó diễn giải |

Ý chính: **tiền xử lý là một quyết định mô hình hoá**, không phải bước "dọn dẹp" trung tính. Nó thay đổi \|V\|, TF, DF, IDF nên thay đổi kết quả tìm kiếm.

**OOV (Out-Of-Vocabulary):** từ trong query/test không có trong từ điển lúc fit → bị bỏ qua (vector = 0 ở chiều đó).

### 1.3. Count vector (Bag-of-Words)

Mỗi văn bản là một vector \|V\| chiều; phần tử thứ $j$ là số lần từ $j$ xuất hiện. Mất hoàn toàn thứ tự từ.

Ví dụ: vocabulary `[cat, dog, eats, fish, likes]`
- D1 = "cat eats fish" → `[1, 0, 1, 1, 0]`
- D2 = "dog eats fish" → `[0, 1, 1, 1, 0]`
- D3 = "cat likes fish" → `[1, 0, 0, 1, 1]`

### 1.4. Công thức TF-IDF

- **Term Frequency:** $tf(t,d) = \dfrac{c(t,d)}{\sum_{t'} c(t',d)}$ (chuẩn hoá theo độ dài văn bản; $\sum_t tf(t,d) = 1$)
- **Document Frequency:** $df(t) = |\{d : t \in d\}|$ — số văn bản chứa $t$
- **Inverse Document Frequency:** $idf(t) = \log\dfrac{N}{df(t)}$
- **Biến thể của scikit-learn (smooth_idf=True):** $idf(t) = \log\dfrac{1+N}{1+df(t)} + 1$
- **TF-IDF:** $tfidf(t,d) = tf(t,d) \times idf(t)$

**Trực giác:** một từ quan trọng với văn bản nếu nó **xuất hiện nhiều trong văn bản đó** nhưng **hiếm trong toàn corpus**.

**Bài tính mẫu** (N = 3, dùng ln):

| Term | df | idf = ln(3/df) |
| :--- | :---: | :---: |
| cat | 2 | ln 1,5 ≈ 0,405 |
| dog | 1 | ln 3 ≈ 1,099 |
| eats | 2 | ≈ 0,405 |
| fish | 3 | ln 1 = **0** |
| likes | 1 | ≈ 1,099 |

D1 = "cat eats fish": tf mỗi từ = 1/3 → TF-IDF: cat ≈ 0,135; eats ≈ 0,135; fish = **0**.
- `fish` có trong mọi văn bản → không phân biệt được văn bản nào → idf = 0 → TF-IDF = 0.
- Vì sao kết quả tự cài (0,2448) khác sklearn (0,5443)? Vì sklearn dùng công thức idf có `+1`: từ phổ biến như `fish` không bị triệt tiêu về 0, và sklearn còn chuẩn hoá L2 mỗi vector. **Các convention khác nhau cho số khác nhau**, cần ghi rõ dùng công thức nào.

### 1.5. Cosine similarity

$$\cos(x,y) = \frac{x^\top y}{\|x\|_2\,\|y\|_2}$$

- Đo **góc** giữa hai vector, không đo độ dài → văn bản dài và ngắn cùng chủ đề vẫn giống nhau.
- Với vector không âm (TF-IDF), cos ∈ [0, 1]; với embedding, cos ∈ [−1, 1].
- Ví dụ: $x=[1,1,1]$, $y=[1,1,0]$ → $\cos = \dfrac{2}{\sqrt3\sqrt2} = \dfrac{2}{\sqrt6} \approx 0{,}816$.
  Không bằng 2/3 vì 2/3 là kiểu Jaccard (giao/hợp); cosine chia cho **trung bình nhân của hai độ dài**, nên y "ngắn" (2 từ) và nằm hoàn toàn trong x được thưởng nhiều hơn.

### 1.6. Sparsity

$$S = 1 - \frac{\text{nnz}}{N \times |V|}$$

Thực nghiệm 30.000 văn bản, \|V\| = 186.582 → **S = 99,935%**. Lý do: mỗi văn bản chỉ dùng vài chục đến vài trăm từ khác nhau trong ~186K chiều. Vì vậy phải lưu bằng ma trận thưa (scipy **CSR**: chỉ lưu giá trị khác 0 + chỉ số cột + con trỏ hàng).

### 1.7. Search engine và đánh giá retrieval

Query được biến đổi bằng **cùng vectorizer** đã fit trên corpus → tính cosine với mọi văn bản → sắp xếp giảm dần → top-k.

| Metric | Công thức | Ý nghĩa |
| :--- | :--- | :--- |
| Precision@k | (số văn bản liên quan trong top-k) / k | Top-k có "sạch" không |
| Recall@k | (số liên quan trong top-k) / (tổng số liên quan) | Có bỏ sót không |
| MRR | $\frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q}$ (rank của kết quả đúng đầu tiên) | Kết quả đúng đầu tiên xuất hiện sớm không |

Kết quả của bạn: trung bình P@5 = 0,20; R@5 = 0,50; MRR = 0,50.

### 1.8. Các failure case kinh điển của lexical matching

1. **Đồng nghĩa (synonymy):** `heart attack treatment` vs `myocardial infarction therapy` → cos = 0,0000.
2. **Đa nghĩa (polysemy):** `transformer` (mô hình) bị lẫn với máy biến áp.
3. **Cụm từ bị bẻ vụn:** `natural language processing` thành 3 unigram rời rạc.
4. **Lặp từ bất thường:** văn bản lặp một từ nhiều lần bị đẩy lên top (TF tăng tuyến tính).

**Hướng khắc phục:**
- **BM25:** TF bão hoà $\frac{tf\,(k_1+1)}{tf + k_1(1-b+b\frac{|d|}{avgdl})}$ (thường $k_1 \in [1{,}2;\,2]$, $b = 0{,}75$) — lặp từ nhiều lần không tăng điểm vô hạn, có chuẩn hoá độ dài.
- **N-gram / phrase query** để giữ cụm từ.
- **Dense retrieval** (Sentence-BERT) hoặc **hybrid** (BM25 + dense) để hiểu đồng nghĩa.

---

## 2. Lab 02 — Mô hình ngôn ngữ N-gram, Smoothing, Perplexity

### 2.1. Language Model là gì

Gán xác suất cho một chuỗi từ, dùng **chain rule**:

$$P(w_1,\dots,w_T) = \prod_{t=1}^{T} P(w_t \mid w_1,\dots,w_{t-1})$$

Ứng dụng: dự đoán từ tiếp theo (bàn phím điện thoại), xếp hạng câu (nhận dạng giọng nói, dịch máy, sửa lỗi chính tả), sinh văn bản.

### 2.2. Giả định Markov và N-gram

Chỉ nhìn $n-1$ từ trước: $P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-n+1:t-1})$

| Mô hình | Xác suất | MLE |
| :--- | :--- | :--- |
| Unigram | $P(w_t)$ | $C(w)/N$ |
| Bigram | $P(w_t \mid w_{t-1})$ | $C(w_{t-1}w_t)/C(w_{t-1})$ |
| Trigram | $P(w_t \mid w_{t-2}w_{t-1})$ | $C(w_{t-2}w_{t-1}w_t)/C(w_{t-2}w_{t-1})$ |

Thực tế thêm token `<s>` (đầu câu) và `</s>` (cuối câu) để mô hình hoá cả việc bắt đầu/kết thúc câu.

**Bài tính** (corpus: `the cat eats fish` / `the cat likes fish` / `the dog eats meat`, N = 12, \|V\| = 7):
- $P(\text{the}) = 3/12 = 0{,}25$
- $P(\text{cat}\mid\text{the}) = 2/3$, $P(\text{dog}\mid\text{the}) = 1/3$, $P(\text{eats}\mid\text{cat}) = 1/2$
- $P(\text{the cat eats fish}) = \frac14\cdot\frac23\cdot\frac12\cdot\frac12 = \frac1{24} \approx 0{,}0417$
- $\sum_w P(w\mid\text{the}) = 1$ vì tử số cộng lại đúng bằng $C(\text{the})$.
- **Thêm một từ vào câu không thể làm xác suất tăng** (nhân thêm một số ≤ 1) → không so sánh trực tiếp $P(S)$ giữa câu dài/ngắn khác nhau, phải chuẩn hoá (perplexity hoặc log-prob trung bình).

### 2.3. Log probability (chống underflow)

Nhân hàng trăm xác suất nhỏ → số dưới giới hạn float → về 0. Dùng log biến tích thành tổng:

$$\log P(S) = \sum_t \log P(w_t \mid h_t)$$

### 2.4. Zero probability và Smoothing

**Vấn đề:** bigram hợp lý nhưng chưa thấy trong train (vd `study AI`) → MLE = 0 → cả câu có P = 0, log P = −∞, PPL = ∞.
Nguyên lý: **"Không quan sát thấy ≠ không thể xảy ra."**

**Laplace (Add-1):**

$$P_{\text{Lap}}(w \mid h) = \frac{C(h,w)+1}{C(h)+|V|}$$

Ví dụ $C(\text{cat})=10$, $|V|=5$:
- $C(\text{cat eats})=0$ → $1/15 \approx 0{,}067$
- $C(\text{cat eats})=3$ → $4/15 \approx 0{,}267$ (MLE là 0,3 → bị chiết khấu)

Smoothing = **lấy bớt xác suất của sự kiện đã thấy chia cho sự kiện chưa thấy**, tổng vẫn = 1.

**Nhược điểm Laplace khi \|V\| lớn:** mẫu số $C(h)+|V|$ với \|V\| = 186.618 làm mọi xác suất rất nhỏ và phân phối quá phẳng. Kết quả nghịch lý trong lab: **Unigram có test PPL (2.050) tốt hơn Bigram (6.703) và Trigram (43.047)**.

**Các kỹ thuật tốt hơn (kiến thức mở rộng):**
- **Add-k** ($k<1$, vd 0,01): bớt "thô bạo" hơn add-1.
- **Interpolation (Jelinek–Mercer):** $\hat P = \lambda_3 P_{tri} + \lambda_2 P_{bi} + \lambda_1 P_{uni}$, $\sum\lambda = 1$, chọn $\lambda$ trên validation.
- **Backoff (Katz, Stupid Backoff):** không có trigram thì lùi về bigram, rồi unigram.
- **Absolute discounting:** trừ một hằng số $d$ (~0,75) từ mọi count.
- **Kneser–Ney:** absolute discounting + unigram dựa trên **số ngữ cảnh khác nhau** mà từ xuất hiện (continuation probability). Ví dụ `Francisco` xuất hiện nhiều nhưng hầu như chỉ sau `San` → không nên được ưu tiên ở ngữ cảnh mới. Đây là phương pháp mạnh nhất cho n-gram.

### 2.5. Perplexity

$$PP(W) = P(w_1\dots w_N)^{-1/N} = \exp\!\Big(-\frac1N\sum_{i=1}^N \ln P(w_i\mid h_i)\Big)$$

- Trực giác: **số lựa chọn trung bình mà mô hình "phân vân"** ở mỗi bước. PPL thấp = ít bối rối = mô hình tốt hơn (trên cùng test set và cùng vocabulary).
- Ví dụ: P = 0,5 · 0,25 · 0,5 = 1/16 → PP = 16^{1/3} ≈ 2,52. Đổi 0,25 thành 0,1 → PP = 40^{1/3} ≈ 3,42. **Một xác suất nhỏ cũng làm PPL tăng mạnh** vì −ln p → ∞ khi p → 0.
- PPL thấp **không** đồng nghĩa văn bản sinh ra hay với con người (có thể lặp từ, vô nghĩa).
- Chỉ so sánh PPL khi cùng tokenizer/vocabulary.

### 2.6. Kết quả và bài học thực nghiệm

| Mô hình | Unique n-grams | Singletons |
| :--- | :---: | :---: |
| Unigram | 186.618 | 47,41% |
| Bigram | 2.739.018 | 71,30% |
| Trigram | 6.404.589 | 86,21% |

| Mô hình | Train PPL | Val PPL | Test PPL |
| :--- | :---: | :---: | :---: |
| Bigram MLE | 209,81 | ∞ | ∞ |
| Trigram MLE | 19,71 | ∞ | ∞ |
| Unigram (Laplace) | 1.915 | 2.279 | **2.051** |
| Bigram Laplace | 5.351 | 8.152 | 6.703 |
| Trigram Laplace | 27.463 | 49.396 | 43.047 |

Bài học:
1. Tăng $n$ → biết thêm ngữ cảnh **nhưng** sparsity tăng theo cấp $|V|^n$.
2. Trigram MLE có train PPL thấp nhất = **học vẹt (overfit)**, ra test thì ∞.
3. \|V\| (số từ đơn) **không đổi** khi tăng $n$; chỉ số n-gram khác nhau tăng.
4. Trigram không nhất thiết tốt hơn bigram trên test (corpus nhỏ, smoothing kém).
5. Context 100 từ: trigram **chỉ dùng 2 từ cuối**, bỏ qua 97 từ đầu → động lực cho RNN/Transformer.

Lỗi next-word prediction đã phân tích: `the cat → s` (tokenizer tách `cat's` + lệch domain), `machine learning → and` (context quá ngắn + stopword tần suất cao + Laplace làm phẳng).

---

## 3. Lab 03 — Word Embeddings

### 3.1. Distributional Hypothesis

> "Words that occur in similar contexts tend to have similar meanings." (Harris 1954; "You shall know a word by the company it keeps" — Firth 1957)

`The doctor treated the patient` / `The physician treated the patient` → `doctor` và `physician` có ngữ cảnh giống nhau → vector nên gần nhau.

Với one-hot hoặc TF-IDF, hai từ khác nhau luôn **trực giao** (cos = 0) → không thể biểu diễn đồng nghĩa.

### 3.2. Ma trận đồng xuất hiện (word–context matrix)

$X \in \mathbb{R}^{|V|\times|V|}$; $X_{ij}$ = số lần từ ngữ cảnh $j$ nằm trong cửa sổ $\pm k$ quanh từ đích $i$.

Corpus `the cat eats fish / the dog eats fish / the cat likes milk / the dog likes meat`, k = 1 (cột: the, cat, dog, eats, likes, fish, milk, meat):

| | the | cat | dog | eats | likes | fish | milk | meat |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| cat | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| dog | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| eats | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
| likes | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

→ vector `cat` và `dog` giống hệt nhau → cos = 1. (Nếu bỏ `the` khỏi vocabulary như trong đề thì cat = dog = [.., eats 1, likes 1, ..].)

Thực nghiệm: \|V\| = 3.043; window 1 → 5 làm số phần tử khác 0 tăng ~4 lần (35.370 → 149.628) nhưng vẫn thưa > 98%.

**Vấn đề:** \|V\| = 100.000 → ma trận 10¹⁰ ô, vẫn thưa, chiều quá lớn, từ phổ biến (`the`) chiếm ưu thế.
**Cải tiến cổ điển (kiến thức mở rộng):**
- **PPMI:** $\text{PMI}(w,c) = \log\frac{P(w,c)}{P(w)P(c)}$, giữ phần dương → giảm ảnh hưởng của từ phổ biến.
- **SVD / LSA:** phân rã $X \approx U\Sigma V^\top$, giữ $d$ thành phần lớn nhất → vector dense $d$ chiều.
- **GloVe:** học vector sao cho $w_i^\top \tilde w_j + b_i + \tilde b_j \approx \log X_{ij}$ — kết hợp thống kê toàn cục và học như mạng nơ-ron.

### 3.3. Sparse vs Dense

| | Sparse (co-occurrence, TF-IDF) | Dense (embedding) |
| :--- | :--- | :--- |
| Số chiều | \|V\| (10⁴–10⁵) | 50–300 |
| Phần tử khác 0 | rất ít | gần như tất cả |
| Ý nghĩa mỗi chiều | một từ cụ thể (diễn giải được) | không diễn giải trực tiếp |
| Đồng nghĩa | không gần nhau | gần nhau |

**Dense không phải lúc nào cũng tốt hơn**: Lab 04 cho thấy TF-IDF thắng Average Word2Vec trên IMDb; tìm kiếm mã sản phẩm, tên riêng, từ khoá chính xác thì lexical tốt hơn.

### 3.4. Word2Vec

Mạng nơ-ron nông (1 lớp ẩn tuyến tính); embedding là **ma trận trọng số** học được qua bài toán dự đoán giả (self-supervised, không cần nhãn).

- **CBOW** (Continuous Bag of Words): **ngữ cảnh → từ đích**. Lấy trung bình embedding các từ ngữ cảnh rồi dự đoán từ giữa. `the ___ eats fish` → `cat`. Nhanh, tốt cho từ phổ biến.
- **Skip-gram:** **từ đích → từng từ ngữ cảnh**. `cat` → `the`, `eats`, `fish`. Chậm hơn, tốt cho từ hiếm và corpus nhỏ.

Hàm mục tiêu Skip-gram: $P(c \mid w) = \dfrac{\exp(u_c^\top v_w)}{\sum_{c'\in V}\exp(u_{c'}^\top v_w)}$.
Mẫu số chạy trên toàn \|V\| → quá đắt → dùng:
- **Negative sampling:** chỉ phân biệt cặp thật (w, c) với k cặp ngẫu nhiên (k = 5–20): $\log\sigma(u_c^\top v_w) + \sum_{i=1}^k \log\sigma(-u_{n_i}^\top v_w)$.
- **Hierarchical softmax:** cây Huffman, chi phí $O(\log|V|)$.
- **Subsampling** từ quá phổ biến (`the`, `of`).

Siêu tham số (gensim): `vector_size` (d), `window`, `min_count`, `sg` (0 = CBOW, 1 = Skip-gram), `negative`, `epochs`.

Cấu hình lab: CBOW, d = 100, window = 5, min_count = 2, 10 epochs, 10.000 văn bản (214.188 câu, 3,66 triệu token) → 53.812 từ, ~17 s.

### 3.5. Ảnh hưởng siêu tham số

- **Window nhỏ (2):** học quan hệ **cú pháp/thay thế** (từ cùng loại, có thể thay nhau: doctor ~ physician).
- **Window lớn (5–10):** học quan hệ **chủ đề/liên kết** (doctor ~ hospital, patient), nhưng thêm nhiễu.
- **Dimension:** 50 → 100 cải thiện; 100 → 300 trên dữ liệu vừa phải không tăng đáng kể mà tăng thời gian (12,8 s → 32,9 s) và RAM (46 → 149 MB). **Số chiều phải tương xứng với lượng dữ liệu.**
- **Corpus nhỏ (100 câu):** doctor/physician không chắc gần nhau — ít lần xuất hiện, không đủ ngữ cảnh chung.

### 3.6. Đánh giá embedding

- **Intrinsic:**
  - Word similarity: so cosine với điểm người chấm (WordSim-353, SimLex-999), dùng tương quan Spearman.
  - Analogy: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$ (tìm từ có cos lớn nhất với vector kết quả, loại 3 từ đầu vào). Lab: ra đúng `queen` ở cả 3 kích thước.
- **Extrinsic:** dùng embedding cho bài toán thật (phân loại, NER…) và đo hiệu năng — Lab 04 chính là đánh giá extrinsic.
- **Trực quan hoá:** PCA / t-SNE xuống 2D.

Bài tính: $v_{doctor}=[0{,}8;0{,}1;0{,}7]$, $v_{physician}=[0{,}7;0{,}2;0{,}8]$, $v_{banana}=[-0{,}2;0{,}9;-0{,}1]$
- cos(doctor, physician) = 1,14 / (1,068 × 1,082) ≈ **0,987**
- cos(doctor, banana) = −0,14 / (1,068 × 0,927) ≈ **−0,141**
- $x=[1,2,1]$, $y=[2,4,2]$ → cos = 1: cùng hướng, khác độ lớn → cosine bỏ qua độ lớn (tần suất), chỉ quan tâm "mẫu hình" ngữ cảnh.

### 3.7. Lỗi và giới hạn của static embedding

Lỗi quan sát được:
- `car`–`automobile` chỉ 0,29: `automobile` hiếm (~25 lần) và chỉ nằm trong cụm cố định (`automobile insurance`).
- `doctor`–`child` 0,69: **thiên kiến chủ đề** của corpus (y tế nhi khoa).
- `banana`–`oregano` 0,81: từ hiếm (14 lần), window rộng gom ngữ cảnh "nấu ăn".

Nguyên nhân chung: tần suất thấp, corpus nhỏ, thiên kiến chủ đề, window rộng. Ngoài ra embedding còn **học cả thiên kiến xã hội** (vd giới tính–nghề nghiệp) từ dữ liệu.

**Giới hạn căn bản — Polysemy:** mỗi từ **một vector duy nhất**. `bank` (ngân hàng) và `bank` (bờ sông) bị ép vào một điểm trung bình hoặc bị nghĩa phổ biến hơn chi phối.

**Mở rộng:**
- **FastText:** vector từ = tổng vector các n-gram ký tự (`<wh`, `whe`, `her`…) → có vector cho từ OOV, tốt với từ hiếm/biến hình.
- **Contextual embeddings (ELMo, BERT):** vector được tính lại theo câu nhờ self-attention → `bank` cạnh `river` khác `bank` cạnh `money`.

---

## 4. Lab 04 — Phân loại văn bản

### 4.1. Bài toán

Dataset $D=\{(x_i,y_i)\}_{i=1}^N$, học $f(x)\to y$. Pipeline:

```text
raw text → preprocessing → representation → classifier → prediction → evaluation → error analysis
```

Các dạng: sentiment (IMDb, SST-2, Yelp), topic (AG News, 20 Newsgroups, DBPedia), spam (SMS Spam), intent/question (TREC-6), NLI (SNLI), paraphrase (QQP).

### 4.2. Kiểm tra dữ liệu trước khi train

Số mẫu, số lớp, **phân phối lớp (imbalance?)**, độ dài văn bản, trùng lặp, thiếu text, thiếu nhãn.
IMDb: 50.000 review, cân bằng 50/50, trung bình > 230 từ, **418 bản trùng** đã loại. Chia 70/15/15 (stratified).

### 4.3. Data leakage

Pipeline đúng:

```text
Dataset → split Train/Val/Test → fit vectorizer CHỈ trên Train → transform Val/Test → train model
```

Fit TF-IDF trên toàn bộ dữ liệu rồi mới chia là **leakage** vì:
1. **Vocabulary leakage:** từ chỉ có trong test đã vào từ điển.
2. **IDF leakage:** $df(t)$ đã đếm cả văn bản test.
→ Kết quả đánh giá lạc quan giả tạo. Các dạng leakage khác: trùng lặp giữa train và test (lý do phải dedup trước), chọn siêu tham số trên test. Dùng `sklearn.pipeline.Pipeline` để tránh.

Vai trò ba tập: **Train** học tham số; **Validation** chọn siêu tham số, early stopping; **Test** chỉ dùng **một lần** cuối cùng.

### 4.4. Logistic Regression

$$z = w^\top x + b,\qquad P(y=1\mid x)=\sigma(z)=\frac1{1+e^{-z}}$$

- Dự đoán: $\hat y = 1$ nếu $P \ge \tau$ (thường $\tau = 0{,}5$, tương đương $z \ge 0$).
- Huấn luyện bằng **binary cross-entropy**: $\mathcal L = -[y\log p + (1-y)\log(1-p)]$, thêm regularization L2 (tham số `C` = 1/λ trong sklearn; C nhỏ → regularize mạnh).
- Đa lớp: **softmax** $P(y=k\mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}$.
- **Diễn giải được:** $w_j > 0$ đẩy về positive, $w_j < 0$ đẩy về negative. Lab: `great` +8,69, `excellent` +7,07; `worst` −10,47, `bad` −8,84, `waste` −7,15.

Bài tính: $x=[1,2,0]$, $w=[0{,}5;-0{,}2;0{,}4]$, $b=0{,}1$ → $z = 0{,}5-0{,}4+0+0{,}1 = 0{,}2$ → $\sigma(0{,}2)\approx 0{,}550$ → positive (τ = 0,5).
$P = 0{,}72$: τ = 0,5 → positive; τ = 0,8 → negative.

**Ngưỡng τ:** tăng τ → dự đoán positive ít hơn, chắc chắn hơn → **Precision ↑, Recall ↓**; giảm τ thì ngược lại.

**Linear SVM:** tìm siêu phẳng có **margin lớn nhất**, hinge loss $\max(0, 1 - y\,z)$; rất hợp với dữ liệu văn bản thưa nhiều chiều. Lab: F1 0,9070 (nhỉnh hơn LR 0,9028). Mô hình tuyến tính + TF-IDF là baseline rất mạnh cho phân loại văn bản.

### 4.5. Metrics

Confusion matrix:

| | Pred Pos | Pred Neg |
| :--- | :---: | :---: |
| **Actual Pos** | TP | FN |
| **Actual Neg** | FP | TN |

- Accuracy $= \frac{TP+TN}{TP+TN+FP+FN}$
- Precision $= \frac{TP}{TP+FP}$ — trong số đoán positive, bao nhiêu đúng (tránh **báo động giả**).
- Recall $= \frac{TP}{TP+FN}$ — trong số positive thật, bắt được bao nhiêu (tránh **bỏ sót**).
- F1 $= \frac{2PR}{P+R}$ — trung bình điều hoà, chỉ cao khi cả P và R cao.
- Đa lớp: **macro-F1** (trung bình đều các lớp, coi trọng lớp nhỏ), **micro-F1** (gộp TP/FP/FN, = accuracy với single-label), **weighted-F1**.

Bài tính (TP 80, FN 20, FP 10, TN 90): P = 80/90 ≈ 0,889; R = 80/100 = 0,80; F1 ≈ 0,842; Acc = 170/200 = 0,85.
Spam: **FP** = thư thật bị vào spam (tệ, mất thư quan trọng) → ưu tiên **Precision**. **FN** = spam lọt vào inbox. Chẩn đoán ung thư, phát hiện mã độc → ưu tiên **Recall**.

**Accuracy trap:** 950 negative / 50 positive, đoán tất cả negative → Accuracy 95% nhưng Recall(pos) = 0, F1 = 0 → vô dụng. Phải xem Precision/Recall/F1 từng lớp, macro-F1, confusion matrix. Với imbalance: `class_weight='balanced'`, over/under-sampling, điều chỉnh threshold, PR-AUC.

### 4.6. Average Word2Vec

$$\vec x = \frac1n\sum_{i=1}^n \vec w_i$$

Văn bản dài (> 230 từ) → các từ cảm xúc mạnh bị **pha loãng** bởi hàng trăm từ trung tính; mất thứ tự từ và phủ định. Kết quả: F1 0,8449, thấp hơn TF-IDF ~6 điểm.
Cải tiến: trung bình có trọng số TF-IDF, SIF; tốt nhất là mô hình chuỗi (Lab 05).

### 4.7. N-gram features và từ phủ định

- `"this movie is not good"`: TF-IDF unigram và Avg W2V đều đoán **positive** vì `good` có trọng số dương lớn, `not` chỉ là một feature độc lập. cos("this movie is good", "this movie is not good") theo Avg W2V = 0,936.
- Thêm bigram (`ngram_range=(1,2)`): `not good` thành feature riêng có trọng số âm → P(pos) giảm từ 63,19% xuống **15,59%** → đúng.
- Đánh đổi: \|V\| tăng 87.734 → 555.050 (với `min_df=2`); không lọc thì gần 1,93 triệu feature, chậm gấp 5 lần mà không tốt hơn. **Tăng vocabulary không chắc tăng accuracy** (thêm nhiễu, overfit).
- **Đừng bỏ stopwords máy móc** trong sentiment: danh sách stopword chứa `not`, `no`, `never`.

| Representation | Classifier | \|V\| | Acc | F1 |
| :--- | :--- | :---: | :---: | :---: |
| TF-IDF uni | LR | 87.734 | 0,9016 | 0,9028 |
| Avg W2V | LR | 100 chiều | 0,8428 | 0,8449 |
| TF-IDF uni | Linear SVM | 87.734 | 0,9060 | 0,9070 |
| TF-IDF uni+bi | LR | 555.050 | **0,9062** | **0,9074** |

### 4.8. Error analysis — các loại lỗi của Bag-of-Words

1. **Mô tả cốt truyện ≠ đánh giá phim:** `evil`, `cowardly`, `shoot` là mô tả phản diện chứ không phải chê phim.
2. **Mixed sentiment:** mở đầu phàn nàn, kết luận khen.
3. **Concessive clause ("khen trước chê sau"):** `amazingly beautiful visual elements but is…`
4. Ngoài ra: sarcasm (mỉa mai), phủ định xa, từ hiếm, thuật ngữ domain.

→ BoW mù với cấu trúc diễn ngôn và liên từ đảo chiều (`but`, `however`) → cần mô hình chuỗi / attention.

---

## 5. Lab 05 — Sequence Models: RNN, LSTM, GRU

> Hạn nộp 21/10/2026. Phần này tổng hợp lý thuyết từ đề W5 và kiến thức nền.

### 5.1. Động lực

Văn bản là **chuỗi có thứ tự**, nhưng TF-IDF và Avg W2V là biểu diễn tĩnh:
- `dog bites man` và `man bites dog` có **cùng** Avg W2V → không phân biệt được.
- Mô hình chuỗi đọc từng token và **cập nhật trạng thái**: $h_t = f(x_t, h_{t-1})$ → $h_t$ tóm tắt mọi thứ đã đọc đến thời điểm $t$.

### 5.2. Pipeline

```text
Raw text → Tokenizer → Vocabulary → Token IDs → Padding/Truncation → Embedding → RNN/LSTM/GRU → h_T → Linear → sigmoid/softmax
```

- **Token IDs:** `"I really like this movie"` → `[12, 47, 31, 8, 102]`. Thường dành ID 0 = `<PAD>`, 1 = `<UNK>` (từ OOV).
- **Padding:** thêm `<PAD>` cho đủ độ dài để gom thành batch: `[12, 47, 31, 8, 102, 0, 0, 0]`. Padding **không phải thông tin ngôn ngữ** → nên dùng `padding_idx=0` trong Embedding, masking hoặc `pack_padded_sequence` (PyTorch) để RNN bỏ qua. Lưu ý: post-padding mà lấy $h_T$ ở vị trí cuối thì RNN đã "đọc" một loạt PAD → nên lấy hidden tại vị trí token thật cuối cùng.
- **Truncation:** cắt chuỗi dài hơn `max_length` → **mất thông tin** (thường là kết luận ở cuối review!). Có thể cắt đầu (pre) hoặc cuối (post).
- **Chọn max_length:** dựa vào **histogram độ dài** (vd percentile 90–95%, median, mean), cân bằng giữa % token giữ lại và chi phí tính toán — không chọn "vì chạy nhanh".
- **Embedding layer:** ma trận $E \in \mathbb{R}^{|V|\times d}$, tra cứu hàng theo ID; khởi tạo ngẫu nhiên hoặc từ Word2Vec/GloVe (Lab 03), có thể freeze hoặc fine-tune.

### 5.3. Simple RNN (Elman)

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$
$$z = W_{hy}h_T + b_y,\qquad P(y\mid x) = \text{softmax}(z)\ \text{(hoặc sigmoid với nhị phân)}$$

- **Cùng một bộ trọng số** dùng lại ở mọi bước (weight sharing) → xử lý chuỗi độ dài bất kỳ.
- $h_0$ thường là vector 0.
- `tanh` giữ giá trị trong (−1, 1), tránh hidden state phình to.
- Số tham số RNN: $h(d + h) + h$ (= $W_{xh}$, $W_{hh}$, $b_h$; PyTorch có hai bias nên là $h(d+h) + 2h$).

**Bài tính một bước:** $x_t=[1,0]^\top$, $h_{t-1}=[0{,}5;0{,}2]^\top$, $W_{xh}=\begin{bmatrix}0{,}5&0{,}2\\0{,}1&0{,}4\end{bmatrix}$, $W_{hh}=\begin{bmatrix}0{,}3&0{,}1\\0{,}2&0{,}5\end{bmatrix}$, $b=0$

- $W_{xh}x_t = [0{,}5;\ 0{,}1]$
- $W_{hh}h_{t-1} = [0{,}3\cdot0{,}5+0{,}1\cdot0{,}2;\ 0{,}2\cdot0{,}5+0{,}5\cdot0{,}2] = [0{,}17;\ 0{,}20]$
- $a_t = [0{,}67;\ 0{,}30]$
- $h_t = \tanh(a_t) \approx [0{,}585;\ 0{,}291]$

**Cài đặt RNN cell (numpy):**

```python
class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        k = 1 / np.sqrt(hidden_size)
        self.W_xh = np.random.uniform(-k, k, (hidden_size, input_size))
        self.W_hh = np.random.uniform(-k, k, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        return np.tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h)
```

Kiểm chứng bằng `torch.nn.RNNCell`: copy `weight_ih` ← `W_xh`, `weight_hh` ← `W_hh`, `bias_ih + bias_hh` ← `b_h`, so `np.allclose`.

**Information flow** (`I did not like this movie`): khi đọc `not`, $h_3$ mã hoá tín hiệu phủ định; khi đọc `like`, $h_4 = f(x_{like}, h_3)$ nên **cùng** từ `like` cho hidden state khác nhau tuỳ có `not` trước hay không → mô hình có thể học "đảo cực tính". Avg W2V thì `not` chỉ cộng một vector vào trung bình, không tương tác với `like`.

**Phân biệt thứ tự** (`dog bites man` vs `man bites dog`): phép cập nhật phi tuyến, không giao hoán → thứ tự đưa vào khác nhau thì $h_T$ khác nhau.

### 5.4. Backpropagation Through Time, vanishing/exploding gradient

Trải RNN theo thời gian thành mạng sâu T lớp rồi lan truyền ngược (**BPTT**):

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}} = \prod_{k} \text{diag}(1-h_k^2)\,W_{hh}$$

- Các thừa số có độ lớn < 1 (đạo hàm tanh ≤ 1, giá trị riêng của $W_{hh}$ nhỏ) → tích → 0: **vanishing gradient** → token ở xa gần như không ảnh hưởng tới việc cập nhật trọng số → **khó học phụ thuộc xa**. Chuỗi càng dài, tích càng nhiều thừa số → càng nặng.
- Các thừa số > 1 → tích → ∞: **exploding gradient** → loss NaN, huấn luyện bất ổn. Khắc phục: **gradient clipping** (`clip_grad_norm_`).
- Khắc phục vanishing: LSTM/GRU (gating), khởi tạo trực giao, ReLU cẩn thận, truncated BPTT, và cuối cùng là attention.
- Lưu ý khi làm lab: một thí nghiệm nhỏ (đặt từ khoá ở đầu/giữa/cuối) là **quan sát thực nghiệm**, không chứng minh được vanishing gradient — đó là **giải thích lý thuyết**.

### 5.5. LSTM

Thêm **cell state** $c_t$ (băng chuyền bộ nhớ) và 3 cổng (sigmoid ∈ (0,1) = "mở bao nhiêu phần trăm"):

| Thành phần | Công thức | Vai trò |
| :--- | :--- | :--- |
| Forget gate | $f_t = \sigma(W_f[x_t,h_{t-1}]+b_f)$ | Giữ bao nhiêu bộ nhớ cũ |
| Input gate | $i_t = \sigma(W_i[x_t,h_{t-1}]+b_i)$ | Ghi bao nhiêu thông tin mới |
| Candidate | $\tilde c_t = \tanh(W_c[x_t,h_{t-1}]+b_c)$ | Nội dung mới đề xuất |
| Cell state | $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ | Cập nhật **cộng** |
| Output gate | $o_t = \sigma(W_o[x_t,h_{t-1}]+b_o)$ | Lộ ra bao nhiêu |
| Hidden | $h_t = o_t\odot\tanh(c_t)$ | Đầu ra mỗi bước |

**Tại sao giảm vanishing:** $\partial c_t/\partial c_{t-1} = f_t$ (cập nhật cộng, không qua nhân ma trận + tanh) → nếu $f_t \approx 1$ gradient đi qua nhiều bước gần như nguyên vẹn ("constant error carousel").

**Bài tính:** $f=0{,}9$, $i=0{,}2$, $\tilde c=0{,}5$, $c_{t-1}=0{,}8$ → $c_t = 0{,}72 + 0{,}10 = 0{,}82$.
- $f_t \to 1$: **giữ** thông tin cũ. $f_t \to 0$: **xoá** bộ nhớ (vd gặp `but` thì quên phần khen trước đó).

Số tham số LSTM ≈ $4\,[h(d+h)+h]$ (gấp 4 lần RNN).

### 5.6. GRU

Đơn giản hơn LSTM: **không có cell state riêng**, 2 cổng:

- Update gate: $z_t = \sigma(W_z[x_t,h_{t-1}])$ — trộn cũ/mới (gộp vai trò forget + input).
- Reset gate: $r_t = \sigma(W_r[x_t,h_{t-1}])$ — bỏ qua bao nhiêu quá khứ khi tạo candidate.
- Candidate: $\tilde h_t = \tanh(W[x_t,\ r_t\odot h_{t-1}])$
- $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t$

Tham số ≈ $3\,[h(d+h)+h]$ → ít hơn LSTM, nhanh hơn, hiệu năng thường tương đương.

| | RNN | LSTM | GRU |
| :--- | :-: | :-: | :-: |
| Recurrent, hidden state | ✓ | ✓ | ✓ |
| Cell state riêng | | ✓ | |
| Gates | | 3 (f, i, o) | 2 (z, r) |
| Tham số (bội số của RNN) | 1× | 4× | 3× |
| Phụ thuộc xa | Hạn chế | Cải thiện | Cải thiện |

**LSTM không chắc thắng RNN trên mọi dataset:** câu ngắn, dữ liệu ít → RNN đủ dùng, LSTM nhiều tham số dễ overfit.

### 5.7. Thiết kế thí nghiệm và đánh giá

- So sánh RNN/LSTM/GRU: **giữ cố định** vocabulary, embedding dim, batch size, optimizer, learning rate, số epoch, split → chỉ đổi kiến trúc (biến kiểm soát).
- Báo cáo ngoài Acc/P/R/F1: **số tham số, thời gian train, số epoch, sequence length, vocabulary size**. Mô hình NLP không chỉ đánh giá bằng độ chính xác.
- **max_length 50/100/200:** đo % token giữ lại, thời gian, validation loss, test F1. Tăng max_length → nhiều ngữ cảnh hơn nhưng chậm hơn (RNN tuần tự O(T)), tốn bộ nhớ (BPTT lưu mọi bước), khó học hơn vì vanishing. **Trade-off context ↔ chi phí.**
- **Truncate về 30 token:** mất phần lớn review, đặc biệt kết luận ở cuối, mất cấu trúc "but…".
- **Learning curves:** train loss ↓ liên tục nhưng val loss ↓ rồi ↑ → **overfitting** (mô hình học thuộc train). Khắc phục: early stopping, dropout, weight decay, giảm kích thước mô hình, thêm dữ liệu. Không chỉ nhìn epoch cuối.
- Error analysis: phân loại lỗi (negation, long-range dependency, sarcasm, ambiguous, rare word, domain term, chuỗi rất dài, truncation, noise) và trả lời **lỗi có phải do sequence modeling không** — đừng mặc định "LSTM không hiểu câu".

Kiến thức mở rộng hay hỏi: **Bidirectional RNN** (đọc cả hai chiều, nối $[\overrightarrow{h};\overleftarrow{h}]$); **stacked RNN** (nhiều tầng); optimizer Adam; loss `BCEWithLogitsLoss`.

### 5.8. Giới hạn của RNN → Attention

1. Muốn dùng $x_1$ khi xử lý $x_{100}$, thông tin phải đi qua ~99 bước → bị nén vào một vector cố định (**bottleneck**) và suy giảm.
2. Tính **tuần tự** → không song song hoá theo thời gian → chậm trên GPU.

**Attention** cho mỗi token "nhìn" trực tiếp mọi token khác:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$

Đường đi giữa hai token bất kỳ = 1 bước, tính song song được → Self-Attention → Transformer → BERT/GPT (contextual embedding giải quyết polysemy của Lab 03).

---

## 6. Bảng so sánh xuyên suốt

| Thuộc tính | TF-IDF + LR | Avg W2V + LR | RNN | LSTM | GRU | Transformer |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| Sparse | ✓ | | | | | |
| Dense | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hiểu đồng nghĩa | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Thứ tự từ | Hạn chế (chỉ qua n-gram) | ✗ | ✓ | ✓ | ✓ | ✓ (positional encoding) |
| Trạng thái phụ thuộc ngữ cảnh | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Phụ thuộc xa | ✗ | ✗ | Hạn chế | Cải thiện | Cải thiện | Tốt |
| Polysemy (1 từ nhiều vector) | ✗ | ✗ | Một phần (qua $h_t$) | Một phần | Một phần | ✓ |
| Chi phí | Thấp | Thấp | Cao hơn | Cao hơn | Cao hơn | Cao, nhưng song song |
| Diễn giải | Dễ (trọng số từ) | Khó | Khó | Khó | Khó | Attention map |

| Biểu diễn | Phụ thuộc ngữ cảnh | Sparse/Dense | Nhiều vector/từ |
| :--- | :-: | :-: | :-: |
| TF-IDF | Không | Sparse | Không |
| Co-occurrence | Không | Sparse | Không |
| Word2Vec / GloVe / FastText | Không | Dense | Không |
| BERT (contextual) | Có | Dense | Có |

---

## 7. Phương pháp làm lab (quy trình khoa học)

Mọi lab đi theo chuỗi:

```text
Theory recap → Calculation (tính tay) → Prediction → Experiment → Implementation
→ Application → Evaluation → Error analysis → Reflection → Limitation → Topic tiếp theo
```

- **Prediction trước experiment** theo mẫu `Prediction / Reason / Confidence`, sau đó đối chiếu. Mục tiêu là kiểm tra khả năng **dự đoán hành vi** của mô hình, không phải tính số.
- **Tự cài từ đầu** (TF-IDF, cosine, n-gram LM, co-occurrence, RNN cell) rồi **so với thư viện** (sklearn, gensim, PyTorch) — chú ý khác biệt convention.
- **Không chỉ báo accuracy**: luôn kèm P/R/F1, confusion matrix, chi phí (thời gian, bộ nhớ, tham số).
- **Error analysis** dựa trên mẫu thật, gắn nguyên nhân cụ thể (preprocessing, sparsity, domain mismatch, thứ tự từ, phủ định…).
- **Phân biệt quan sát thực nghiệm và giải thích lý thuyết.**
- **AI statement**: Tool / Purpose / Generated / Modified / Verified. AI không được thay thế prediction, calculation, giải thích cơ chế, reflection.

---

## 8. Ngân hàng câu hỏi vấn đáp

### Lab 01
1. **Tại sao TF-IDF sparse?** Chiều = \|V\| (~186K), mỗi văn bản chỉ dùng vài trăm từ → > 99,9% là 0.
2. **Từ có mặt ở hầu hết văn bản có IDF thấp?** $df \approx N$ → $\log(N/df)\approx 0$; từ phổ biến không giúp phân biệt văn bản.
3. **IDF cao nhưng TF-IDF thấp?** Văn bản không chứa từ đó → TF = 0. TF-IDF cao cần cả hiếm trong corpus **và** nhiều trong văn bản.
4. **Tại sao cosine hợp với document vector?** Đo hướng, chuẩn hoá L2 loại bỏ ảnh hưởng độ dài văn bản.
5. **Tại sao preprocessing đổi kết quả search?** Thay đổi \|V\|, TF, DF, IDF, OOV.
6. **Failure case & hướng tiếp theo?** Đồng nghĩa (`heart attack` vs `myocardial infarction`, cos = 0) → cần dense/semantic embedding.

### Lab 02
1. **Vì sao bigram có zero probability?** Cặp từ hợp lý chưa xuất hiện trong train → count = 0.
2. **Vì sao dùng log probability?** Tránh underflow; tích → tổng.
3. **Perplexity thấp nghĩa là gì?** Mô hình ít bất ngờ, gán xác suất cao hơn cho dữ liệu thật.
4. **Trigram không chắc tốt hơn bigram trên test?** Sparsity nặng hơn, nhiều trigram mới bị smoothing phạt.
5. **`cat eats` chưa thấy?** MLE → 0; Laplace → $1/(C(cat)+|V|)$.
6. **Context 100 từ, trigram dùng được 97 từ đầu không?** Không, chỉ 2 từ cuối.
7. **PPL thấp hơn có luôn sinh văn bản tốt hơn?** Không.

### Lab 03
1. **Distributional hypothesis?** Từ có ngữ cảnh giống nhau thì nghĩa giống nhau.
2. **Tại sao doctor và physician gần nhau?** Cùng ngữ cảnh (treat, patient, prescribe…).
3. **CBOW vs Skip-gram?** Ngữ cảnh → từ đích vs từ đích → ngữ cảnh. Skip-gram tốt cho từ hiếm.
4. **Window lớn tốt hay xấu?** Học chủ đề rộng nhưng mất tính cú pháp, thêm nhiễu.
5. **Tại sao Word2Vec không phân biệt hai nghĩa của bank?** Static: 1 từ = 1 vector.
6. **Tại sao TF-IDF không phải word embedding?** Biểu diễn văn bản, sparse, các từ trực giao, không mã hoá tương đồng nghĩa.
7. **Dense có luôn tốt hơn?** Không (Lab 04: TF-IDF thắng Avg W2V).

### Lab 04
1. **Accuracy 95% chưa chắc tốt?** Accuracy trap với dữ liệu mất cân bằng; xem Recall/F1.
2. **TF-IDF vs Avg W2V?** Sparse/lexical/giữ sức nặng từ khoá vs dense/semantic/bị pha loãng.
3. **"this movie is not good" gây lỗi?** BoW mất thứ tự, `good` trọng số dương lấn `not`.
4. **Leakage khi fit TF-IDF trước khi split?** Vocabulary và IDF đã thấy test.
5. **Precision vs Recall, khi nào ưu tiên?** Spam → Precision; bệnh/mã độc → Recall.
6. **Bigram cải thiện sentiment vì sao?** Bắt ngữ cảnh cục bộ, `not good` thành feature âm riêng.
7. **Thay đổi threshold?** τ ↑ → P ↑, R ↓.

### Lab 05
1. **RNN khác Avg W2V ở điểm nào?** RNN xử lý tuần tự, $h_t$ phụ thuộc thứ tự và tương tác giữa các từ; Avg W2V là trung bình không thứ tự.
2. **Hidden state $h_t$ là gì?** Vector tóm tắt toàn bộ thông tin từ $x_1$ đến $x_t$ — "bộ nhớ" của mạng.
3. **Tại sao RNN bị vanishing gradient?** BPTT nhân nhiều Jacobian có độ lớn < 1 → gradient về 0 với bước xa.
4. **Forget gate dùng làm gì?** Quyết định giữ/xoá bao nhiêu phần cell state cũ.
5. **Tăng sequence length làm train chậm?** RNN tính tuần tự O(T), BPTT lưu mọi bước, nhiều padding.
6. **GRU khác LSTM?** Không có cell state riêng, 2 cổng (update, reset), ít tham số hơn (~3× vs 4× RNN).
7. **Train loss giảm, val loss tăng?** Overfitting → early stopping, dropout, regularization.
8. **Truncate 30 token mất gì?** Phần lớn nội dung, đặc biệt kết luận cuối review và các đoạn đảo chiều sau `but`.
9. **Padding có phải thông tin ngôn ngữ?** Không; cần mask/`padding_idx`/pack để mô hình bỏ qua.

---

## 9. Bảng công thức tra nhanh

| Khái niệm | Công thức |
| :--- | :--- |
| TF | $c(t,d)/\sum_{t'}c(t',d)$ |
| IDF | $\log(N/df(t))$; sklearn: $\log\frac{1+N}{1+df}+1$ |
| TF-IDF | $tf \times idf$ |
| Cosine | $x^\top y / (\|x\|\|y\|)$ |
| Sparsity | $1 - nnz/(N\cdot|V|)$ |
| MRR | $\frac1{|Q|}\sum 1/\text{rank}$ |
| Chain rule | $P(w_{1:T})=\prod_t P(w_t\mid w_{1:t-1})$ |
| Bigram MLE | $C(w_{t-1}w_t)/C(w_{t-1})$ |
| Laplace | $(C(h,w)+1)/(C(h)+|V|)$ |
| Perplexity | $\exp(-\frac1N\sum\ln P(w_i\mid h_i))$ |
| PMI | $\log\frac{P(w,c)}{P(w)P(c)}$ |
| Skip-gram softmax | $\exp(u_c^\top v_w)/\sum_{c'}\exp(u_{c'}^\top v_w)$ |
| Analogy | $\vec b - \vec a + \vec c \approx \vec d$ |
| Avg embedding | $\frac1n\sum_i\vec w_i$ |
| Logistic | $\sigma(w^\top x + b)$, $\sigma(z)=1/(1+e^{-z})$ |
| BCE loss | $-[y\log p+(1-y)\log(1-p)]$ |
| Precision / Recall | $TP/(TP+FP)$ / $TP/(TP+FN)$ |
| F1 | $2PR/(P+R)$ |
| RNN | $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b)$ |
| LSTM cell | $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$, $h_t=o_t\odot\tanh(c_t)$ |
| GRU | $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
