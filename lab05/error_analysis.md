# Báo cáo Phân tích Lỗi Thực nghiệm (Error Analysis) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — 2026  
**Mô hình đánh giá:** Gated Recurrent Unit (GRU) trên tập Test IMDb (7,438 mẫu)  
**Tập siêu tham số:** `vocab_size=20,000`, `embed_dim=64`, `hidden_dim=64`, `seq_len=100`, `batch_size=256`, `epochs=10`

---

## 1. Tổng quan ma trận nhầm lẫn và phân phối lỗi

Trên toàn bộ 7,438 mẫu của tập Test:
- **True Positives (TP):** 3,284
- **True Negatives (TN):** 2,912
- **False Positives (FP):** 790
- **False Negatives (FN):** 452
- **Accuracy:** 83.30% | **F1-Score:** 84.09%

Mô hình GRU có tỷ lệ dự đoán thiên về Positive (FP = 790 > FN = 452). Để hiểu sâu sắc bản chất cơ chế của sequence model (mô hình hồi quy tuần tự) và những giới hạn vốn có của kiến trúc RNN/GRU khi xử lý văn bản thực tế, chúng tôi trích xuất và phân tích 6 mẫu đại diện (3 mẫu đoán đúng và 3 mẫu đoán sai).

---

## 2. Bảng tổng hợp 6 mẫu phân tích thực tế

| Index | Văn bản (Snippet) | Độ dài (Từ) | True Label | Pred Label | $P(\text{Positive})$ | Phân loại cơ chế lỗi |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **#0** | *"This was truly a deeply moving movie in every sense of the word. I myself was a Mormon missionary..."* | 159 | **positive** | **positive** | 0.9216 | Dự đoán đúng (Đồng nhất cảm xúc, từ khóa mạnh) |
| **#1** | *"Okay, here's the deal. There's this American pilot who's flying along, minding his own business..."* | 443 | **positive** | **positive** | 0.7158 | Dự đoán đúng (Giọng điệu phiêu lưu / tường thuật) |
| **#2** | *"It's nice to see a film with real people with honest feelings. Sissy Spacek is so absolutely convincing..."* | 76 | **positive** | **positive** | 0.9856 | Dự đoán đúng (Độ dài ngắn, mật độ cảm xúc cao) |
| **#4** | *"This movie can't decide what it is -- a soft porno or a sf movie. Not enough plot for a real movie..."* | 208 | **negative** | **positive** | 0.5676 | Lỗi Cắt cụt (Truncation Loss) & Nghịch ngữ |
| **#7** | *"This is a movie about a black man buying a airline company and turning the company into a African-centric..."* | 191 | **negative** | **positive** | 0.5995 | Lỗi Trì hoãn cảm xúc (Delayed Sentiment Resolution) |
| **#10** | *"Luther the Geek (1990) is a dull horror movie and is really bad even by Troma's standards!!..."* | 211 | **negative** | **positive** | 0.7542 | Lỗi Châm biếm (Irony) & Nhượng bộ (Contrastive Praise) |

---

## 3. Phân tích chi tiết từng trường hợp

### 3.1. Các mẫu dự đoán ĐÚNG (Correct Classifications)

#### Mẫu #0 (Index 0 — True: Positive | Pred: Positive, $p = 0.9216$)
* **Toàn văn trích dẫn:**
  > *"This was truly a deeply moving movie in every sense of the word. I myself was a Mormon missionary and I know first hand the wanting to complete my mission but at the same time hiding the fact that I was gay. Like the character Aaron, I was sent home for being found out and excommunicated... This movie really took me back to those days and helped me to realize, years later, how fortunate I was to have a family that accepted me... I was truly moved to tears. I highly recommend this movie to all who read this review and also declare it a must buy."*
* **Đặc điểm ngôn ngữ:**
  - Độ dài 159 từ. Cảm xúc tích cực duy trì xuyên suốt từ đầu tới cuối câu.
  - Các cụm từ tích cực xuất hiện ngay tại 20 token đầu tiên: *"deeply moving movie"*, *"in every sense of the word"*.
  - Đoạn kết luận lặp lại các tín hiệu mạnh: *"applauding"*, *"moved to tears"*, *"highly recommend"*, *"must buy"*.
* **Cơ chế mô hình:**
  Cổng cập nhật (Update Gate $z_t$) của GRU nhận diện các vector từ mang cảm xúc tích cực ở đầu câu, mở cổng nạp vào trạng thái ẩn $h_t$ và duy trì thông tin này qua suốt quá trình duyệt chuỗi. Dù câu chuyện có các từ ngữ tiêu cực về hoàn cảnh cá nhân (*"excommunicated", "shame"*), nhưng cấu trúc cú pháp nhấn mạnh việc được gia đình chấp nhận (*"fortunate to have a family that accepted me"*) giúp GRU duy trì xác suất tích cực áp đảo ($p = 0.9216$).

---

#### Mẫu #1 (Index 1 — True: Positive | Pred: Positive, $p = 0.7158$)
* **Toàn văn trích dẫn:**
  > *"Okay, here's the deal. There's this American pilot who's flying along, minding his own business, when suddenly he's outnumbered by evil, cowardly non-American fighter planes... Now this American pilot was doing nothing wrong, but those evil non-Americans didn't care and before you know it he's banged up in a foreign jail and sentenced to death!!... I cannot recommend this film enough. It was the first ever videocassette movie that I brought..."*
* **Đặc điểm ngôn ngữ:**
  - Bài viết rất dài (443 từ). Người viết dành 90% dung lượng để kể lại nội dung phim theo lối hài hước, châm biếm nhân vật phản diện (*"evil, cowardly", "foreign jail", "sentenced to death", "massacre"*).
  - Lời khen thực sự chỉ xuất hiện ở đoạn cuối: *"I cannot recommend this film enough"*.
* **Cơ chế mô hình:**
  Do mô hình áp dụng `max_length = 100`, đoạn cuối bài đã bị cắt bỏ hoàn toàn! Tuy nhiên, mô hình vẫn dự đoán đúng nhãn Positive với $p = 0.7158$. Nguyên nhân là ở 100 từ đầu tiên, các từ mang sắc thái tiêu cực chỉ mô tả kẻ thù trong kịch bản chứ không dùng tính từ chê bai phim (*"terrible", "boring", "awful"*). Sự kết hợp của giọng văn kể chuyện nhịp điệu nhanh giúp trạng thái ẩn $h_T$ không bị rơi vào vùng tiêu cực.

---

#### Mẫu #2 (Index 2 — True: Positive | Pred: Positive, $p = 0.9856$)
* **Toàn văn trích dẫn:**
  > *"It's nice to see a film with real people with honest feelings. Sissy Spacek is so absolutely convincing as a simple, yet nice, daughter to Robert Farnsworth, who finally, in his last role, gets to show what a fine actor he was. It is hard to believe that this is a David Lynch film. It is slow and even, sweet and moving. One of the best unless you like car chases, sex scenes, and violence."*
* **Đặc điểm ngôn ngữ:**
  - Văn bản ngắn (76 từ), nằm trọn vẹn trong `max_length = 100`, không hề bị cắt cụt (Zero Truncation Loss).
  - Mật độ tính từ khen ngợi dày đặc: *"nice to see"*, *"honest feelings"*, *"absolutely convincing"*, *"fine actor"*, *"sweet and moving"*, *"one of the best"*.
* **Cơ chế mô hình:**
  Mô hình GRU không gặp bất kỳ trở ngại nào về vanishing gradient hay padding noise. Trạng thái ẩn $h_t$ tích lũy liên tục các kích hoạt dương qua từng bước thời gian, đạt độ tự tin gần như tuyệt đối ($p = 0.9856$).

---

### 3.2. Các mẫu dự đoán SAI (Misclassifications)

#### Mẫu #4 (Index 4 — True: Negative | Pred: Positive, $p = 0.5676$)
* **Toàn văn trích dẫn:**
  > *"This movie can't decide what it is -- a soft porno or a sf movie. Not enough plot for a real movie, but way too much for a porno.<br /><br />In a galaxy far far away, a good princess (we know she's good because she has long blonde hair, wears a white toga, is polite, and is mooning wistfully about her dying mother when we first see her) inherits the kingdom over her elder sister, who is evil... It really goes downhill from there. The acting is ludicrous, the dialog sounds like it was written for a porno, and the general storyline is ridiculous. There's not even enough skin to make happy the sort of person who'd watch this for skin. I love 'good' bad movies, and this one's not even enjoyable on a 'bad movie' level."*
* **Nguyên nhân gốc rễ (Root Cause): Lỗi Cắt cụt chuỗi (Truncation Loss) & Mô tả cốt truyện đối lập**
  1. Bài review dài 208 từ. Với `max_length = 100`, mô hình chỉ đọc được đoạn mở đầu tóm tắt kịch bản.
  2. Trong 100 từ đầu tiên, người viết dùng từ ngữ tích cực để chế giễu nhân vật: *"a good princess"*, *"she's good because"*, *"polite"*, *"white toga"*.
  3. Lời phê bình thật sự đanh thép nằm hoàn toàn ở nửa sau (từ từ 110 trở đi): *"acting is ludicrous"*, *"storyline is ridiculous"*, *"not even enjoyable"*. Việc cắt cụt sau 100 từ đã vứt bỏ toàn bộ phần kết luận phê phán, khiến GRU bị đánh lừa bởi các từ tích cực giả tạo ở nửa đầu.
* **Bài học:** Khẳng định giới hạn của việc giới hạn `max_length`. Cần chiến lược trích xuất thông minh hơn (như giữ 50 từ đầu + 50 từ cuối) hoặc cơ chế Attention để bao quát toàn văn bản.

---

#### Mẫu #7 (Index 7 — True: Negative | Pred: Positive, $p = 0.5995$)
* **Toàn văn trích dẫn:**
  > *"This is a movie about a black man buying a airline company and turning the company into a African-centric over the top airliner. They even portray the owner as not only being in control of the airline, but also controlling part of the air terminal at the airport. One day this guy wins $100 million dollars a the next time you see him, he is walking all over the airport acting like the owner of the airport. Everyone calls this movie a parody, but nothing about this movie shouts parody! This movie is a flop and will forever be in the $4.95 bin at Wal-Mart.<br /><br />I can't even come to terms to why MGM would waste 16 million dollars on this movie. This movie doesn't even warrant straight-to-video status... Don't waste your time at the video store..."*
* **Nguyên nhân gốc rễ (Root Cause): Hiện tượng trì hoãn cảm xúc (Delayed Sentiment Resolution)**
  1. Người viết mở đầu hoàn toàn trung tính để thuật lại tiền đề kịch bản (*"buying an airline", "wins 100 million dollars", "calls this movie a parody"*). Các từ này kích hoạt thiên hướng tích cực nhẹ trong không gian nhúng ($p = 0.5995$).
  2. Đánh giá tiêu cực thực sự (*"flop"*, *"waste 16 million dollars"*, *"humiliates me"*, *"don't waste your time"*) xuất hiện muộn ở cuối bài, vượt quá cửa sổ 100 từ.
* **Bài học:** Đối với dạng bài review phim, người dùng có xu hướng nêu kết luận ở câu cuối (conclusion). Xử lý chuỗi một chiều bị giới hạn nghiêm trọng nếu không đọc được tới cuối đoạn văn.

---

#### Mẫu #10 (Index 10 — True: Negative | Pred: Positive, $p = 0.7542$)
* **Toàn văn trích dẫn:**
  > *"Luther the Geek (1990) is a dull horror movie and is really bad even by Troma's standards!! It's about a freak who bites the heads off chickens, and kills people by biting their necks out... Luther gets released from prison after 25 yrs because apparentely he's been a 'model' prisoner and deserves another chance in life... Then we get treated to a long winded and boring movie, with awful acting all the way through... I love low budget horror movies and i think 'The Toxic Avenger' is an all time classic, but i'm afraid Luther the Geek is just boring, illogical and dull... 3/10"*
* **Nguyên nhân gốc rễ (Root Cause): Châm biếm (Irony), Cấu trúc nhượng bộ (Contrastive Praise) & Trộn lẫn cảm xúc**
  1. Tác giả sử dụng biện pháp mỉa mai khi gọi kẻ giết người là *"a 'model' prisoner and deserves another chance in life"* — mô hình chuỗi ngây thơ tiếp nhận cụm *"model prisoner / deserves another chance"* như một ngữ cảnh nhân văn tích cực.
  2. Tác giả thể hiện sự yêu thích với dòng phim kinh dị hạng B: *"I love low budget horror movies and i think 'The Toxic Avenger' is an all time classic"* — từ *"love"* và *"all time classic"* tạo ra kích hoạt tích cực cực mạnh.
  3. Dù có xuất hiện các từ chê (*"dull", "bad"*), nhưng cấu trúc cú pháp phức tạp đan xen giữa tình yêu thể loại và sự chê bai cụ thể đối với bộ phim này đã vượt quá khả năng phân giải ngữ nghĩa của mạng GRU đơn tầng 64 chiều.

---

## 4. Kết luận và Đề xuất cải tiến

| Loại lỗi chính | Tỷ lệ ảnh hưởng | Cơ chế gây lỗi trong Sequence Model | Giải pháp khắc phục |
| :--- | :---: | :--- | :--- |
| **Cắt cụt độ dài (Truncation Loss)** | **~50%** | Ngưỡng `max_length = 100` loại bỏ câu kết luận chốt hạ ở cuối bài review. | Áp dụng chiến lược cắt cụt hỗn hợp (Head + Tail truncation) hoặc tăng `max_length` lên 200–300. |
| **Châm biếm / Mỉa mai (Sarcasm/Irony)** | **~25%** | Nghĩa đen của từ vựng tích cực đối nghịch hoàn toàn với dụng ý tiêu cực của người viết. | Cần mô hình ngôn ngữ lớn (Pre-trained LM) được tiền huấn luyện trên khối lượng dữ liệu khổng lồ. |
| **Cấu trúc nhượng bộ (Contrastive Clauses)** | **~25%** | Các cụm khen ngợi đối tượng phụ (*"I love horror", "classic"*) lấn át từ chê đối tượng chính. | Cơ chế **Self-Attention** (Mục tiêu của LAB 06/07) để gán trọng số chú ý trực tiếp vào từ ngữ nói về bộ phim chính thay vì bối cảnh phụ. |

