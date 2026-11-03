# Báo cáo Phân tích Lỗi Thực nghiệm (Error Analysis) — LAB 07

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Mô hình phân tích:** Transformer Encoder Classifier (4 Attention Heads, Masked Mean Pooling)  
**Tập dữ liệu:** IMDb Test Split (2,500 mẫu đánh giá)  

---

## 1. Tổng quan Ma trận Nhầm lẫn & Phân phối Lỗi

Từ kết quả thực nghiệm trong `experiments.ipynb` (đồ thị `confusion_matrix.png`):
- **Accuracy:** $74.44\%$
- **F1-Score:** $0.7240$
- **Precision:** $0.7795$ | **Recall:** $0.6758$

Mô hình có xu hướng dự đoán thận trọng ở nhãn Positive (Recall thấp hơn Precision), dẫn đến số lượng False Negatives (mẫu thực tế là Positive nhưng bị gán nhãn Negative) chiếm tỷ trọng đáng kể. 

Dưới đây là bảng tổng hợp 6 mẫu đại diện được trích xuất trực tiếp từ tập Test (3 mẫu dự đoán đúng và 3 mẫu dự đoán sai) nhằm phân tích cơ chế hành vi của Transformer.

---

## 2. Bảng tổng hợp 6 Mẫu Thực tế từ Tập Test

| Index | Văn bản trích đoạn (Snippet) | Độ dài | True | Pred | $P(\text{Pos})$ | Cơ chế ngôn ngữ & Nguyên nhân |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **#2** | *"It's nice to see a film with real people with honest feelings. Sissy Spacek is so absolutely convincing..."* | 76 từ | **pos** | **pos** | **0.8994** | **Đúng:** Mật độ từ khóa tích cực cao (*"nice", "honest", "convincing", "fine"*), câu ngắn. |
| **#3** | *"Buddy is an entertaining family film set in a time when 'humanizing' animals... Based on a true story..."* | 114 từ | **pos** | **pos** | **0.8803** | **Đúng:** Chủ đề rõ ràng (*"entertaining", "great love"*), ít từ gây nhiễu. |
| **#4** | *"This movie can't decide what it is -- a soft porno or a sf movie. Not enough plot for a real movie..."* | 208 từ | **neg** | **neg** | **0.1318** | **Đúng:** Cụm từ phủ định và chê bai dồn dập ngay đầu câu (*"can't decide", "not enough plot"*). |
| **#0** | *"This was truly a deeply moving movie in every sense of the word. I myself was a Mormon missionary and I know first hand..."* | 159 từ | **pos** | **neg** | **0.4504** | **Sai:** Từ vựng miền tiêu cực ngoại cảnh (*"hiding", "excommunicated", "found out"*), tự sự cá nhân. |
| **#1** | *"Okay, here's the deal. There's this American pilot who's flying along... when suddenly he's outnumbered by evil, cowardly..."* | 443 từ | **pos** | **neg** | **0.1326** | **Sai:** Truncation Loss (cắt sau 128 từ) + Trào phúng / Từ ngữ tiêu cực trong tóm tắt cốt truyện (*"evil", "cowardly", "shoot down"*). |
| **#5** | *"This is my favorite movie of all time. I just love all the trouble all the teams get into! Its great... so stereotypical..."* | 52 từ | **pos** | **neg** | **0.4588** | **Sai:** Từ mang nghĩa bề mặt tiêu cực trong ngữ cảnh tích cực (*"trouble", "stereotypical"*), câu ngắn phân vân ($p \approx 0.46$). |

---

## 3. Phân tích Chi tiết Từng Trường hợp

### 3.1. Các mẫu dự đoán ĐÚNG (Correct Predictions)

#### Mẫu #2 (Index 2 — True: Positive | Pred: Positive, $p = 0.8994$)
- **Nội dung:** *"It's nice to see a film with real people with honest feelings. Sissy Spacek is so absolutely convincing as a simple, yet nice, daughter to Robert Farnsworth, who finally, in his last role, gets to show what a fine actor he was..."*
- **Đặc điểm ngôn ngữ:** Văn bản có độ dài vừa phải (76 từ, nằm trọn vẹn trong `max_seq_len = 128`), không bị cắt cụt. Mật độ từ ngữ mang cảm xúc tích cực rất dày đặc và phân bổ đều khắp câu: *"nice"*, *"honest"*, *"convincing"*, *"fine"*, *"sweet"*, *"moving"*.
- **Hành vi Attention:** Các attention heads dễ dàng dồn trọng số vào các cặp từ mô tả chất lượng diễn xuất (*"convincing" $\leftrightarrow$ "Spacek"*, *"fine" $\leftrightarrow$ "actor"*). Dù có từ tương phản *"yet"*, vế sau vẫn mang nghĩa tích cực (*"simple, yet nice"*), không gây xung đột ngữ nghĩa.

#### Mẫu #3 (Index 3 — True: Positive | Pred: Positive, $p = 0.8803$)
- **Nội dung:** *"Buddy is an entertaining family film set in a time when 'humanizing' animals... shows the great love that the main characters have for animals and for each other..."*
- **Đặc điểm ngôn ngữ:** Cấu trúc câu khẳng định trực tiếp ngay từ mở đầu (*"entertaining family film"*, *"great love"*). 
- **Hành vi Attention:** Head 1 và Head 4 bắt được liên kết giữa chủ ngữ *"film"* và tính từ đánh giá *"entertaining"*, giúp điểm xác suất Positive lên tới $0.88$.

#### Mẫu #4 (Index 4 — True: Negative | Pred: Negative, $p = 0.1318$)
- **Nội dung:** *"This movie can't decide what it is -- a soft porno or a sf movie. Not enough plot for a real movie, but way too much for a porno..."*
- **Đặc điểm ngôn ngữ:** Phủ định (Negation) kết hợp phê bình trực diện (*"can't decide"*, *"not enough plot"*).
- **Hành vi Attention:** Nhờ có Positional Encoding và cơ chế Self-Attention đa chiều, Transformer nhận diện được phạm vi của từ phủ định *"not"* gắn chặt với *"enough plot"*, không bị đánh lừa bởi từ *"good princess"* xuất hiện ở vế sau mang tính châm biếm.

---

### 3.2. Các mẫu dự đoán SAI (Incorrect Predictions)

#### Mẫu #0 (Index 0 — True: Positive | Pred: Negative, $p = 0.4504$)
- **Nội dung:** *"This was truly a deeply moving movie in every sense of the word. I myself was a Mormon missionary and I know first hand the wanting to complete my mission but at the same time hiding the fact that I was gay. Like the character Aaron, I was sent home for being found out and excommunicated..."*
- **Nguyên nhân lỗi:** 
  1. **Nhiễu từ vựng miền (Domain Lexical Noise):** Người viết chia sẻ câu chuyện đời tư đầy trắc trở gắn liền với nội dung phim. Đoạn văn chứa hàng loạt từ vựng thường gắn với ngữ cảnh tiêu cực trong tập dữ liệu huấn luyện: *"hiding"*, *"excommunicated"*, *"sent home"*, *"found out"*.
  2. **Trọng số phân tán:** Dù câu mở đầu rất khen ngợi (*"deeply moving movie"*), nhưng phần lớn các token tiếp theo lại kể về biến cố cá nhân. Phép Masked Mean Pooling lấy trung bình trên toàn bộ 128 tokens khiến tín hiệu khen ngợi ban đầu bị pha loãng bởi hàng loạt từ vựng có trọng số tiêu cực. Xác suất ra $p = 0.4504$ (rất gần ngưỡng 0.5), cho thấy mô hình ở trạng thái phân vân cao.

#### Mẫu #1 (Index 1 — True: Positive | Pred: Negative, $p = 0.1326$)
- **Nội dung:** *"Okay, here's the deal. There's this American pilot who's flying along, minding his own business, when suddenly he's outnumbered by evil, cowardly non-American fighter planes... who proceed to shoot him down..."*
- **Nguyên nhân lỗi:**
  1. **Tổn thất cắt cụt (Truncation Loss):** Đánh giá này dài tới 443 từ. Do mô hình giới hạn `max_seq_len = 128`, hơn 70% nội dung phía sau (nơi người viết chốt lại đánh giá tích cực và khen bộ phim hấp dẫn) đã bị loại bỏ hoàn toàn.
  2. **Từ ngữ miêu tả phản diện (Plot Description vs Review Sentiment):** Trong 128 từ đầu tiên, tác giả chỉ mải miết tóm tắt kịch bản phim hành động, chứa các từ có sắc thái tiêu cực mạnh: *"evil"*, *"cowardly"*, *"shoot down"*, *"outnumbered"*. Mô hình không thể phân biệt giữa việc "bộ phim có nhân vật phản diện tàn ác" với "người viết ghét bộ phim".

#### Mẫu #5 (Index 5 — True: Positive | Pred: Negative, $p = 0.4588$)
- **Nội dung:** *"This is my favorite movie of all time. I just love all the trouble all the teams get into! Its great. Michael J. Fox looked so young.. but then again i guess he was! The teams are so stereotypical you just have to laugh."*
- **Nguyên nhân lỗi:**
  1. **Nghĩa ngữ cảnh đặc thù (Context-dependent Semantics):** Cụm từ *"love all the trouble"* thể hiện sự yêu thích những tình huống hài hước dở khóc dở cười của nhân vật. Tuy nhiên từ *"trouble"* và *"stereotypical"* trong từ điển thường mang trọng số âm.
  2. **Thói quen viết tắt / Informal text:** *"Its great"* (viết thiếu dấu nháy `'`) và cấu trúc câu vụn vặt khiến biểu diễn phân lớp không dồn trọn vẹn điểm số về Positive.

---

## 4. Phân tích Trọng tâm: "Attention Weights" và Cảnh báo Phương pháp luận

### 4.1. Token quan trọng có xuất hiện trong Attention Pattern không?
Qua việc phân tích ma trận chú ý (ở Mục 20 `attention_heads.png`), các token mang cảm xúc mạnh như *"fantastic"*, *"not"*, *"disappointing"* thường nhận được trọng số chú ý cao từ một hoặc hai head chuyên biệt. Điều này chứng minh Self-Attention có khả năng phát hiện các vùng thông tin giàu tín hiệu.

### 4.2. Cảnh báo quan trọng: Phân biệt "Model Behavior" và "Explanation"
Đề cương LAB 07 yêu cầu sinh viên bắt buộc phải nắm vững nguyên lý:
> **Attention Weights $\neq$ Explanation (Trọng số chú ý không đồng nghĩa với lời giải thích nhân quả của con người).**

Lý do:
1. **Attention chỉ là một thành phần trung gian:** Attention weights $A$ chỉ quyết định cách tổ hợp tuyến tính các Value vectors $V$ ($Z = A V$). Sau bước attention, biểu diễn còn phải đi qua:
   $$\text{Residual Connection} \longrightarrow \text{LayerNorm} \longrightarrow \text{Feed-Forward Network (FFN)} \longrightarrow \text{LayerNorm} \longrightarrow \text{Pooling} \longrightarrow \text{Linear Classifier}$$
   Đặc biệt, tầng FFN áp dụng các phép biến đổi phi tuyến tính mạnh mẽ (ReLU), có thể khuếch đại hoặc triệt tiêu hoàn toàn các đặc trưng mà attention đã gom tụ.
2. **Hiện tượng Attention Phân tán & Redundancy:** Một token có attention weight cao có thể chỉ đơn giản đóng vai trò làm mỏ neo ngữ pháp (như dấu câu hoặc hư từ `<PAD>`, *"the"*), hoặc do tính chất của ma trận chiếu $W^Q, W^K$.
3. **Kết luận:** Việc nhìn vào heatmap của Attention chỉ cho ta biết **hành vi của mô hình (Model Behavior)** — cụ thể là mô hình đang phân phối trọng số tính toán ở đâu — chứ không thể coi đó là **lời giải thích đáng tin cậy của con người (Human-interpretable Rationale)** về lý do mô hình đưa ra quyết định cuối cùng.

