# Error Analysis — Lab 04

Tài liệu này thực hiện phân tích định tính chuyên sâu các trường hợp dự đoán đúng và sai của mô hình phân loại văn bản Baseline (TF-IDF + Logistic Regression) trên tập dữ liệu kiểm thử (Test Set) của IMDb, tuân thủ đúng yêu cầu tại **Mục 21 của đề bài W4.pdf**.

---

## 1. Bảng tổng hợp các ca phân tích

| STT | Phân loại | Nhãn thực tế (True) | Dự đoán (Pred) | Đoạn trích văn bản (Snippet) | Nguyên nhân xác định |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | **Đúng (Case 1)** | `pos` | `pos` | *"This was truly a deeply moving movie in every sense of the word..."* | Từ khóa cảm xúc trực tiếp, phân cực dương tính rõ rệt. |
| **2** | **Đúng (Case 2)** | `pos` | `pos` | *"It's nice to see a film with real people with honest feelings. Sissy Spacek is so absolutely convincing..."* | Tập hợp các tính từ/trạng từ khen ngợi tích cực phân bổ đều đặn. |
| **3** | **Đúng (Case 3)** | `pos` | `pos` | *"Buddy is an entertaining family film set in a time when 'humanizing' animals... Buddy shows the great love..."* | Ngữ cảnh thể loại phim gia đình, từ vựng mang sắc thái ấm áp, tích cực. |
| **4** | **Sai (Case 4)** | `pos` | `neg` | *"Okay, here's the deal. There's this American pilot... suddenly he's outnumbered by evil, cowardly non-American fighter planes... who proceed to shoot him down..."* | **Lexical Ambiguity & Domain Plot Description:** Tóm tắt tình tiết phim hành động chứa nhiều từ tiêu cực. |
| **5** | **Sai (Case 5)** | `pos` | `neg` | *"If in the 90's you're adapting a book written in the 50's, set the bloody thing in the 50's... It's a simple rule that Hollywood habitually keeps violating. And that's the problem..."* | **Mixed Sentiment & Frustrated Tone:** Phê phán khâu bối cảnh ở đầu bài lấn át lời khen ở phần sau. |
| **6** | **Sai (Case 6)** | `neg` | `pos` | *"THE KING MAKER will doubtless be a success in Thailand where the similar (but superior) 'The Legend of Suriyothai' set box office records... has some amazingly beautiful visual elements but is..."* | **Concessive Clause ("Khen trước chê sau"):** Lời khen hình thức và dự đoán doanh thu lừa mô hình BoW. |

---

## 2. Phân tích chi tiết từng trường hợp

### 2.1. Nhóm dự đoán đúng (Correctly Classified)

#### Case 1: Cảm xúc chân thực với các từ khóa phân cực mạnh
- **True Label:** `pos` | **Predicted:** `pos`
- **Text:** *"This was truly a deeply moving movie in every sense of the word. I myself was a Mormon missionary and I know first hand the wanting to complete my mission but at the same time hiding the fact that I was gay..."*
- **Phân tích:** 
  - Người đánh giá bày tỏ sự xúc động sâu sắc bằng các cụm từ có trọng số dương rất lớn: `deeply moving` ($w > 0$), `truly` ($w > 0$).
  - Bài viết không dùng cấu trúc câu phức tạp hay phủ định ngầm. Vector TF-IDF của văn bản tích lũy các giá trị dương áp đảo, giúp hàm sigmoid cho ra xác suất $P(\text{pos}) > 0.95$.

#### Case 2: Nhiều tính từ khen ngợi chất lượng diễn xuất
- **True Label:** `pos` | **Predicted:** `pos`
- **Text:** *"It's nice to see a film with real people with honest feelings. Sissy Spacek is so absolutely convincing as a simple, yet nice, daughter to Robert Farnsworth, who finally, in his last role, gets to show what a fine actor he was..."*
- **Phân tích:**
  - Văn bản chứa một chuỗi các từ khen ngợi xuất hiện liên tiếp: `nice` ($w = +2.41$), `honest`, `convincing` ($w = +2.82$), `fine` ($w = +2.15$).
  - Mô hình tuyến tính chỉ cần cộng dồn các trọng số dương này vào logit $z = \mathbf{w}^T \mathbf{x} + b$ là đủ vượt xa ngưỡng $\tau = 0.5$.

#### Case 3: Thuật ngữ thể loại và sắc thái ấm áp
- **True Label:** `pos` | **Predicted:** `pos`
- **Text:** *"Buddy is an entertaining family film set in a time when 'humanizing' animals, and making them cute was an accepted way to get people to be interested in them... Buddy shows the great love that the main characters have for animals..."*
- **Phân tích:**
  - Các đặc trưng `entertaining` ($w = +3.81$), `family`, `cute`, `great love` ($w_{\text{great}} = +8.69$) đóng vai trò quyết định.
  - Phim thuộc thể loại thân thiện cho gia đình thường có trường từ vựng rất rõ ràng, ít bị nhiễu bởi các từ ngữ tiêu cực hay mỉa mai.

---

### 2.2. Nhóm dự đoán sai (Incorrectly Classified)

#### Case 4: Nhầm lẫn giữa nội dung cốt truyện (Plot) và cảm nhận cá nhân
- **True Label:** `pos` | **Predicted:** `neg`
- **Text:** *"Okay, here's the deal. There's this American pilot who's flying along, minding his own business, when suddenly he's outnumbered by evil, cowardly non-American fighter planes... who proceed to shoot him down..."*
- **Phân loại lỗi:** **Lexical Ambiguity & Domain-Specific Plot Description**
- **Phân tích nguyên nhân:**
  - Reviewer rất thích bộ phim hành động kịch tính này, nhưng phần lớn đoạn mở đầu được dành để kể lại tình tiết gay cấn của phim.
  - Các từ xuất hiện: `evil` ($w = -3.42$), `cowardly` ($w = -2.11$), `shoot` ($w = -1.95$), `down`. 
  - Mô hình Bag-of-Words (TF-IDF) không phân biệt được đâu là "từ dùng để tả nhân vật phản diện trong kịch bản" và đâu là "từ thể hiện thái độ của người xem đối với chất lượng bộ phim". Sự áp đảo của các từ tiêu cực trong mô tả tình huống khiến mô hình dự đoán nhầm thành `neg`.

#### Case 5: Giọng điệu bực bội và cảm xúc hỗn hợp (Mixed Sentiment)
- **True Label:** `pos` | **Predicted:** `neg`
- **Text:** *"If in the 90's you're adapting a book written in the 50's, set the bloody thing in the 50's and not the '90's. See, 40 year old mores and values tend not to play as well, or ring as true, that far down the road. It's a simple rule that Hollywood habitually keeps violating. And that's the problem with..."*
- **Phân loại lỗi:** **Mixed Sentiment & Stylistic Tone Bias**
- **Phân tích nguyên nhân:**
  - Người viết mở đầu bài viết bằng một lời phàn nàn gay gắt về thói quen chuyển thể bối cảnh của Hollywood: `bloody` ($w = -1.82$), `violating` ($w = -2.10$), `problem` ($w = -3.12$).
  - Mặc dù kết luận chung của bài viết là tích cực (khen ngợi diễn xuất và hình ảnh bất chấp bối cảnh), mật độ các từ ngữ gay gắt ở nửa đầu đã kéo tổng logit $z$ xuống miền âm ($z < 0$). Mô hình không có cơ chế chú ý (attention) để nhận biết rằng người viết vẫn khuyên khán giả nên xem phim.

#### Case 6: Cấu trúc nhượng bộ ("Khen trước chê sau" / Concessive Clause)
- **True Label:** `neg` | **Predicted:** `pos`
- **Text:** *"THE KING MAKER will doubtless be a success in Thailand where the similar (but superior) 'The Legend of Suriyothai' set box office records. The film directed by Lek Kitaparaporn after a screenplay by Sean Casey based on historical fact in 1547 Siam has some amazingly beautiful visual elements but is..."*
- **Phân loại lỗi:** **Concessive Clause, Lexical Deception & Negation Scope**
- **Phân tích nguyên nhân:**
  - Đây là lỗi kinh điển của mô hình Bag-of-Words. Người đánh giá sử dụng kỹ thuật viết lịch sự: mở đầu bằng việc công nhận những mặt tốt về kỹ thuật hình ảnh và tiềm năng phòng vé (`success` $w = +2.54$, `superior` $w = +2.10$, `records`, `amazingly beautiful` $w_{\text{amazing}} = +4.73$).
  - Tuy nhiên, sau liên từ đảo chiều `but is...`, người viết bắt đầu chê tơi tả kịch bản rời rạc và diễn xuất đơ cứng. 
  - Do các từ ngữ khen ngợi hình thức ở đoạn đầu có trọng số TF-IDF rất mạnh, mô hình Bag-of-Words bị "đánh lừa" và dự đoán thành `pos`, bỏ qua cấu trúc nhượng bộ then chốt của câu.

---

## 3. Tổng kết bài học rút ra từ Error Analysis

1. **Giới hạn của Bag-of-Words:** BoW hoàn toàn mù mờ trước cấu trúc diễn ngôn (discourse structure) như *mở bài phàn nàn - thân bài khen* hoặc *khen hình thức - chê nội dung*.
2. **Nhiễu từ việc tóm tắt cốt truyện:** Trong phim ảnh, tiểu thuyết, việc mô tả nhân vật xấu (*evil, killer, murder*) thường bị mô hình nhầm lẫn là đánh giá chê bai bộ phim.
3. **Cần mô hình ngữ cảnh sâu (Contextual / Sequence Models):** Để giải quyết triệt để các lỗi này, ta cần đến các kiến trúc có khả năng theo dõi mạch văn bản dài và liên từ phụ thuộc như LSTM hoặc Transformer với cơ chế Self-Attention (sẽ tiếp cận ở Lab 05 trở đi).

