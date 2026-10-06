# Lời giải các bài tập tính toán lý thuyết (Calculations)

Tài liệu này trình bày chi tiết các bài tính toán tay lý thuyết theo Mục 6, 7, 8, 16 và 23 của đề bài W3.pdf.

---

## Bài 1: Xây dựng ma trận và vector đồng xuất hiện bằng tay (Mục 6)

### Ngữ liệu (Corpus):
- Câu 1: the cat eats fish
- Câu 2: the dog eats fish
- Câu 3: the cat likes milk
- Câu 4: the dog likes meat

Từ vựng quan sát (Vocabulary): `cat`, `dog`, `eats`, `likes`, `fish`, `milk`, `meat`.  
Cửa sổ ngữ cảnh: $k = 1$ (1 từ liền trước và 1 từ liền sau, bỏ qua từ "the").

### Phân tích ngữ cảnh xuất hiện:
- Với từ `cat`:
  - Trong câu 1: sau `the`, trước `eats` $\to$ context: `eats` (+1)
  - Trong câu 3: sau `the`, trước `likes` $\to$ context: `likes` (+1)
  - Vector của `cat`: $[0, 0, 1, 1, 0, 0, 0]$

- Với từ `dog`:
  - Trong câu 2: sau `the`, trước `eats` $\to$ context: `eats` (+1)
  - Trong câu 4: sau `the`, trước `likes` $\to$ context: `likes` (+1)
  - Vector của `dog`: $[0, 0, 1, 1, 0, 0, 0]$

- Với từ `eats`:
  - Trong câu 1: sau `cat`, trước `fish` $\to$ context: `cat` (+1), `fish` (+1)
  - Trong câu 2: sau `dog`, trước `fish` $\to$ context: `dog` (+1), `fish` (+1)
  - Vector của `eats`: $[1, 1, 0, 0, 2, 0, 0]$

- Với từ `likes`:
  - Trong câu 3: sau `cat`, trước `milk` $\to$ context: `cat` (+1), `milk` (+1)
  - Trong câu 4: sau `dog`, trước `meat` $\to$ context: `dog` (+1), `meat` (+1)
  - Vector của `likes`: $[1, 1, 0, 0, 0, 1, 1]$

Nhận xét: Vector của `cat` và `dog` giống nhau hoàn toàn vì chúng chia sẻ cùng ngữ cảnh xuất hiện (đều đứng trước `eats` và `likes`).

---

## Bài 2: Tính Cosine Similarity giữa hai vector cùng hướng (Mục 6)

Cho hai vector:
- $x = [1, 2, 1]$
- $y = [2, 4, 2]$

Tích vô hướng:
$$x \cdot y = (1 \times 2) + (2 \times 4) + (1 \times 2) = 2 + 8 + 2 = 12$$

Độ dài (Norm) của từng vector:
$$\|x\| = \sqrt{1^2 + 2^2 + 1^2} = \sqrt{6}$$
$$\|y\| = \sqrt{2^2 + 4^2 + 2^2} = \sqrt{4 + 16 + 4} = \sqrt{24} = 2\sqrt{6}$$

Độ tương đồng Cosine:
$$\cos(x, y) = \frac{x \cdot y}{\|x\| \cdot \|y\|} = \frac{12}{\sqrt{6} \cdot 2\sqrt{6}} = \frac{12}{12} = 1.0$$

Ý nghĩa: Hai vector có độ lớn khác nhau ($y = 2x$) nhưng Cosine Similarity bằng 1 thể hiện rằng chúng cùng chỉ về một hướng trong không gian. Độ đo Cosine không phụ thuộc vào độ dài (tần số xuất hiện tuyệt đối) mà chỉ phản ánh tỷ lệ phân bố giữa các chiều ngữ nghĩa.

---

## Bài 3: So sánh Semantic Similarity giữa các từ (Mục 7)

Cho các vector biểu diễn:
- $v_{doctor} = [0.8, 0.1, 0.7]$
- $v_{physician} = [0.7, 0.2, 0.8]$
- $v_{banana} = [-0.2, 0.9, -0.1]$

### 1. Tính $\cos(doctor, physician)$:
- $v_{doctor} \cdot v_{physician} = (0.8 \times 0.7) + (0.1 \times 0.2) + (0.7 \times 0.8) = 0.56 + 0.02 + 0.56 = 1.14$
- $\|v_{doctor}\| = \sqrt{0.8^2 + 0.1^2 + 0.7^2} = \sqrt{0.64 + 0.01 + 0.49} = \sqrt{1.14} \approx 1.0677$
- $\|v_{physician}\| = \sqrt{0.7^2 + 0.2^2 + 0.8^2} = \sqrt{0.49 + 0.04 + 0.64} = \sqrt{1.17} \approx 1.0817$
- $\cos(doctor, physician) = \frac{1.14}{1.0677 \times 1.0817} \approx \frac{1.14}{1.1549} \approx 0.9871$

### 2. Tính $\cos(doctor, banana)$:
- $v_{doctor} \cdot v_{banana} = (0.8 \times -0.2) + (0.1 \times 0.9) + (0.7 \times -0.1) = -0.16 + 0.09 - 0.07 = -0.14$
- $\|v_{banana}\| = \sqrt{(-0.2)^2 + 0.9^2 + (-0.1)^2} = \sqrt{0.04 + 0.81 + 0.01} = \sqrt{0.86} \approx 0.9274$
- $\cos(doctor, banana) = \frac{-0.14}{1.0677 \times 0.9274} \approx \frac{-0.14}{0.9902} \approx -0.1414$

Kết luận: $\cos(doctor, physician) \approx 0.9871$ rất cao, trong khi $\cos(doctor, banana) \approx -0.1414$ âm, phản ánh chính xác doctor gần nghĩa với physician và hoàn toàn không liên quan đến banana.

---

## Bài 4: So sánh biểu diễn Sparse và Dense (Mục 8)

Cho từ vựng $V = 10.000$:
- Biểu diễn A: 10.000 chiều nhưng chỉ có 30 giá trị khác 0.
- Biểu diễn B: 300 chiều và hầu hết các giá trị đều khác 0.

Trả lời các câu hỏi:
1. Biểu diễn nào sparse? Biểu diễn A là sparse (tỷ lệ phần tử 0 chiếm $99.7\%$).
2. Biểu diễn nào dense? Biểu diễn B là dense (kích thước nhỏ gọn 300 chiều, hầu hết các phần tử đều chứa thông tin số thực khác 0).
3. Vì sao dense representation thuận lợi hơn cho semantic similarity?
   - Biểu diễn dense nén các đặc trưng phân phối vào một không gian số chiều thấp liên tục.
   - Tránh hiện tượng trực giao giữa các từ đồng nghĩa (trong biểu diễn sparse, hai từ đồng nghĩa nếu không cùng xuất hiện trong một ngữ cảnh hẹp thì tích vô hướng vẫn bằng 0).
   - Giảm đáng kể chi phí tính toán và bộ nhớ khi áp dụng vào các mô hình học sâu phía sau.
4. Dense representation có chắc chắn tốt hơn trong mọi bài toán không?
   - Không. Trong một số bài toán như tìm kiếm từ khóa chính xác (lexical matching, keyword retrieval), biểu diễn sparse như TF-IDF hay BM25 vẫn hoạt động rất hiệu quả, có khả năng giải thích cao và không đòi hỏi chi phí huấn luyện mô hình nơ-ron phức tạp.

---

## Bài 16: Tập huấn luyện cho CBOW và Skip-gram (Mục 16)

Cho câu: `the cat eats fish` với cửa sổ $window = 1$.

### Mô hình CBOW (Context $\to$ Target):
- Ví dụ 1: Context `[the, eats]` $\to$ Target: `cat`
- Ví dụ 2: Context `[cat, fish]` $\to$ Target: `eats`

### Mô hình Skip-gram (Target $\to$ Context):
- Ví dụ 1 (từ tâm `cat`):
  - Target: `cat` $\to$ Context: `the`
  - Target: `cat` $\to$ Context: `eats`
- Ví dụ 2 (từ tâm `eats`):
  - Target: `eats` $\to$ Context: `cat`
  - Target: `eats` $\to$ Context: `fish`

---

## Bài 23: Bài tập tính toán Word Analogy (Mục 23)

Cho các vector giả định:
- $v_{king} = [8, 2, 7]$
- $v_{man} = [5, 1, 5]$
- $v_{woman} = [5, 3, 5]$

Tính $v_{result} = v_{king} - v_{man} + v_{woman}$:
$$v_{result} = [8 - 5 + 5, \; 2 - 1 + 3, \; 7 - 5 + 5] = [8, 4, 7]$$

Giải thích loại quan hệ:
- Vector hiệu $v_{woman} - v_{man} = [0, 2, 0]$ mã hóa thuộc tính giới tính nữ (nữ quyền, giống cái) mà không làm thay đổi các đặc trưng khác.
- Khi cộng vector giới tính này vào $v_{king}$, vector mới giữ nguyên đặc trưng quyền lực hoàng gia ($8$ và $7$) nhưng thay đổi chiều thuộc tính giới tính từ $2$ lên $4$.
- Vector kết quả $[8, 4, 7]$ đại diện cho vị vua phái nữ, tương ứng với khái niệm `queen` (Nữ hoàng).
