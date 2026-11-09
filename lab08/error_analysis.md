# Báo cáo Phân tích Lỗi Thực nghiệm (Error Analysis) — LAB 08

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Học kỳ:** Học kỳ I — 2026  
**Mô hình phân tích:** Pretrained BERT (`bert-base-uncased`) Fine-Tuned  
**Tập dữ liệu:** IMDb Test Split (1,000 mẫu đánh giá)  

---

## 1. Tổng quan Kết quả Thực nghiệm & Ma trận Nhầm lẫn

Mô hình BERT fine-tuning đạt các chỉ số trên tập kiểm tra IMDb (chi tiết xem tại `confusion_matrix.png`):
- **Accuracy:** $86.00\%$
- **Macro F1-Score:** $0.8649$
- **Precision (Positive):** $0.8390$
- **Recall (Positive):** $0.8924$

Mô hình nhận diện tương đối nhạy với nhãn Positive (Recall đạt $\approx 89.2\%$). Tuy nhiên, trong 1,000 mẫu test vẫn xuất hiện 140 trường hợp dự đoán sai (gồm 86 mẫu False Positive và 54 mẫu False Negative). 

Đặc biệt, việc phân tích file `error_samples.csv` chỉ ra rằng rất nhiều mẫu bị mô hình dự đoán sai với **mức độ tự tin cực kỳ cao (Confidence $> 99.5\%$)**.

---

## 2. Bảng phân tích chi tiết các mẫu lỗi thực tế (Mục 20)

Dưới đây là 8 mẫu lỗi tiêu biểu được trích xuất trực tiếp từ `error_samples.csv` (gồm 3 False Positives, 3 False Negatives và 2 High-confidence mistakes):

| Nhóm | ID | Trích đoạn văn bản (Text snippet) | Độ dài (tokens) | True | Pred | Conf | Phân loại lỗi (Error Type) | Giả thuyết nguyên nhân (Hypothesis) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **FP** | `#460` | *"The story of a woman (Ann) on her death bed... First of all the good. The period detail in the movie is great. The dresses, hair, cars... quality acting... Vanessa Redgrave is quite good... Claire Danes is beautiful... Now for the bad, which unfortunately is everything else..."* | 711 | **neg** | **pos** | **0.9973** | **Long-range dependency / Truncation** | Review dài 711 tokens nhưng mô hình bị giới hạn `max_length=128`. Khoảng 100 token đầu tiên tác giả dành toàn bộ lời khen cho trang phục và diễn viên (*"great", "quality", "good", "beautiful"*). Phần chê bai cốt truyện và kết luận tiêu cực nằm ở nửa sau bị cắt bỏ hoàn toàn, khiến BERT chỉ thấy các tín hiệu tích cực. |
| **FP** | `#434` | *"The figure of empress Elizabeth of Austria... mostly associated with Romy Schneider and the Sissi trilogy (1950s) where beauty, gentleness, sweetness... a powerful portrayal... However, Jean Daniel Verhaeghe's film, though made 50 years later, appears to be a wrong depiction..."* | 1067 | **neg** | **pos** | **0.9968** | **Context comparison / Truncation** | Tác giả mở đầu bằng việc ca ngợi vẻ đẹp kinh điển của bộ phim Sissi năm 1950 (*"beauty", "gentleness", "sweetness", "powerful"*), sau đó mới dùng phim đó làm thước đo để chê bai bộ phim làm lại năm 2004. Do bị cắt sau 128 token, mô hình nhầm tưởng lời khen tác phẩm cũ là đánh giá dành cho bộ phim hiện tại. |
| **FP** | `#968` | *"This film is the freshman effort of Stephanie Beaton and her new production company. While it suffers from a few problems, as every low budget production does... the passion is evident..."* | 259 | **neg** | **pos** | **0.9952** | **Ambiguous sentiment / Polite tone** | Người viết sử dụng văn phong lịch sự, động viên nỗ lực của một đạo diễn trẻ mới vào nghề. Mặc dù tổng thể chấm điểm thấp vì phim kinh phí thấp còn nhiều hạt sạn, nhưng các từ ngữ biểu đạt lại nhẹ nhàng và mang tính khuyến khích (*"passion is evident", "freshman effort"*). |
| **FN** | `#750` | *"Yes this a B- grade horror. But at least the producers, directors, and cast does not pretend this flick is manna from heaven. The plot is corny, a psychotic killer... blood, gore... but it delivers exactly what fans want..."* | 265 | **pos** | **neg** | **0.9967** | **Domain-specific expression / B-movie trope** | Trong dòng phim kinh dị hạng B (B-movie), các đặc tính như *"corny plot"*, *"psychotic killer"*, *"blood, gore"* lại là những điểm hấp dẫn mà người hâm mộ tìm kiếm. BERT bắt gặp dày đặc các từ vựng mang ngữ nghĩa bề mặt tiêu cực này nên tự tin dự đoán nhãn Negative. |
| **FN** | `#462` | *"I've recently watched this movie... we have a lot of fun! This movie is a masterpiece of trash. It is so utterly terrible that you cannot stop laughing..."* | 163 | **pos** | **neg** | **0.9953** | **Sarcasm / Oxymoron (Nghịch hợp)** | Cụm từ nghịch hợp *"masterpiece of trash"* (kiệt tác rác rưởi) thể hiện sự thích thú tột độ khi xem một bộ phim hài ngớ ngẩn ("so bad it's good"). Tầng self-attention của BERT bị chi phối bởi các từ tiêu cực nặng như *"trash", "terrible"*, bỏ sót hàm ý khen ngợi trào phúng. |
| **FN** | `#748` | *"Peter Jacksons version(s) are better films overall... That being said, they are not my favorite... Ralph Bakshi LOTR has a superior atmosphere... overall voice acting... I'll give it a score of 8½ out of 10."* | 1327 | **pos** | **neg** | **0.9953** | **Mixed sentiment & Comparative critique** | Bài viết so sánh đa chiều giữa hai phiên bản Lord of the Rings. Tác giả liệt kê rất nhiều khuyết điểm kỹ xảo của bản 1978 xen kẽ lời phê bình bản của Peter Jackson, tạo ra mật độ từ phản biện cao. Tuy nhiên điểm số cuối cùng là 8.5/10 (tích cực). |
| **High Conf** | `#598` | *"I'm grateful to Cesar Montano in reviving the Visayan film... Downside: the corny and stilted acting of the American antagonist... lack of authenticity in costume... Without those small details, Panaghoy would've been 9/10..."* | 246 | **pos** | **neg** | **0.9952** | **Negation & Flaw listing** | Người viết khen ngợi sự phục hưng điện ảnh vùng miền nhưng dành phần lớn dung lượng liệt kê các chi tiết vụn vặt chưa ưng ý (*"corny", "stilted acting", "lack of authenticity"*). Sự lấn át của các tính từ chê bai cục bộ khiến mô hình tự tin phán đoán sai. |
| **High Conf** | `#62` | *"Okay, this may not be the most sophisticated movie... It's puerile, adolescent, inoffensive, idiotic..And utterly hilarious. Basically, Richie and Eddie run the worst hotel in Britain... hilarious antics..."* | 218 | **pos** | **neg** | **0.9950** | **Self-deprecating comedy tropes** | Thể loại hài nhảm (slapstick comedy) sử dụng hàng loạt tính từ thoạt nhìn mang tính thóa mạ (*"puerile", "adolescent", "idiotic", "worst hotel"*) để ca ngợi độ hài hước (*"utterly hilarious"*). BERT không phân biệt được phong cách hài tự trào này. |

---

## 3. Phân tích Độ tin cậy (Confidence Analysis — Mục 21)

Quan sát đồ thị `confidence_analysis.png`:
1. **Phân phối độ tin cậy:** Phần lớn các dự đoán của mô hình tập trung ở nhóm **High Confidence ($> 0.8$)**. Điều này phản ánh đặc tính của hàm kích hoạt Softmax ở đầu ra mô hình Pretrained: mô hình thường đẩy giá trị logit của nhãn thắng cuộc lên rất cao, tạo ra các xác suất tiệm cận $1.0$.
2. **Hiện tượng $\text{Confidence} \neq \text{Correctness}$ (Tự tin cao không đồng nghĩa với đoán đúng):**
   - Đáng chú ý, toàn bộ 8 mẫu lỗi phân tích ở bảng trên đều có độ tin cậy $P > 0.995$. Mô hình hoàn toàn "chắc chắn" vào quyết định của mình nhưng thực tế lại sai lệch $180^\circ$.
   - **Bản chất:** Confidence của mạng nơ-ron chỉ phản ánh biên độ logit toán học qua hàm Softmax đối với phân phối từ vựng mà nó đã học, chứ **không phải là sự thấu hiểu thực sự về chân lý ngữ nghĩa**. Khi gặp các cấu trúc nghịch lý (sarcasm, review dài bị cắt đoạn đầu, từ ngữ tiêu cực trong thể loại đặc thù), mô hình bị đánh lừa bởi các từ vựng bề mặt có trọng số lớn và đưa ra dự đoán sai lệch với độ tự tin cực đoan (overconfident errors).

---

## 4. Trả lời các câu hỏi bắt buộc (Mục 28: Q1 — Q6)

### Q1: Model nào hưởng lợi nhiều hơn khi training data ít?
* **Trả lời:** **Pretrained Model (BERT)** hưởng lợi vượt trội hơn rất nhiều.
* **Chứng cứ & Giải thích:** 
  - Tại Thí nghiệm 3 (`data_efficiency.png`), khi chỉ có $10\%$ dữ liệu huấn luyện ($400$ mẫu), Pretrained BERT vẫn đạt độ chính xác **$82.60\%$** (chỉ giảm $3.4\%$ so với khi dùng $100\%$ dữ liệu là $86.00\%$).
  - Trong khi đó, mô hình Transformer from scratch trên $10\%$ dữ liệu bị sụp đổ hoàn toàn về mức **$50.20\%$** (tương đương đoán mò ngẫu nhiên). Pretrained model đã mang sẵn tri thức ngôn ngữ từ hàng tỷ từ vựng, nó chỉ cần rất ít mẫu để tìm ra siêu phẳng phân loại phù hợp (tính hiệu quả mẫu cao).

### Q2: Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
* **Trả lời:** Điều đó chứng minh **sức mạnh của việc học biểu diễn ngôn ngữ tổng quát không giám sát (Self-Supervised Pretraining) thông qua Masked Language Modeling (MLM)**.
* **Chứng cứ & Giải thích:**
  - Kết quả Exp 1 cho thấy Pretrained BERT đạt $86.00\%$ Accuracy và $0.8649$ F1, trong khi From-scratch chỉ đạt $49.80\%$.
  - Điều này chứng minh rằng một kiến trúc Transformer nếu chỉ khởi tạo ngẫu nhiên thì không thể tự học được ngữ nghĩa và ngữ pháp phong phú chỉ từ vài nghìn mẫu văn bản của bài toán downstream. Việc pre-train trước trên kho ngữ liệu khổng lồ cung cấp một điểm xuất phát trọng số (initialization) tối ưu, giúp mô hình chuyển giao tri thức (transfer learning) thành công.

### Q3: Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
* **Trả lời:** Điều đó có ý nghĩa rằng **biểu diễn trích xuất tại `[CLS]` của mô hình Pretrained đã chứa đựng gần như đầy đủ các đặc trưng phân biệt cần thiết cho bài toán downstream**.
* **Chứng cứ & Giải thích:**
  - Ở Exp 2, mô hình Frozen Encoder (chỉ huấn luyện 1 tầng Linear với 1,538 tham số) vẫn đạt Accuracy **$77.90\%$** và F1 **$0.7877$** (chỉ kém Full Fine-tuning $8.1\%$).
  - Điều này chứng tỏ vector `[CLS]` sinh ra từ pre-training đã phân cụm ngữ nghĩa tương đối tốt, đủ để một bộ phân loại tuyến tính đơn giản tìm được ranh giới phân tách mà không cần phải thay đổi các tầng sâu bên trong mô hình.

### Q4: Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
* **Trả lời:** Điều đó gợi ý rằng **nhiệm vụ downstream có sự khác biệt nhất định so với mục tiêu pre-training tổng quát, đòi hỏi các cơ chế chú ý phải được tái điều chỉnh (domain/task adaptation)**.
* **Chứng cứ & Giải thích:**
  - Fine-tuning giúp tăng từ $77.90\%$ lên $86.00\%$ (+8.1%). 
  - Điều này cho thấy khi mở khóa toàn bộ tham số, gradient lan truyền ngược giúp các ma trận Self-Attention căn chỉnh lại để tập trung cao độ vào các từ ngữ giàu sắc thái đánh giá phim ảnh (khen/chê), biến các biểu diễn ngôn ngữ tổng quát thành các biểu diễn chuyên biệt hóa cao cho bài toán phân loại cảm xúc.

### Q5: Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
* **Trả lời:** Nguyên nhân chính là **hiện tượng Dịch chuyển miền (Domain Shift) về phân phối từ vựng, phong cách diễn đạt và ngữ cảnh văn bản**.
* **Chứng cứ & Giải thích:**
  - Mỗi miền dữ liệu có những từ vựng chuyên ngành, cấu trúc ngữ pháp và sắc thái biểu đạt cảm xúc rất khác nhau (ví dụ: review phim ảnh dùng từ ngữ nghệ thuật như *"acting, cinematography, plot"*, còn review đồ gia dụng dùng *"durability, battery life, customer service"*).
  - Pretrained model tuy đã thấy nhiều từ trong pre-training, nhưng khi fine-tune trên Domain A, các trọng số đã bị thiên lệch (overfit) vào cách biểu đạt của Domain A, khiến nó mất đi tính khái quát khi đánh giá Domain B.

### Q6: Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
* **Trả lời:** **KHÔNG NÊN tiếp tục tăng sequence length.**
* **Chứng cứ & Giải thích từ LAB 06–07:**
  - Ở Exp 4, khi tăng từ $T=128$ lên $T=256$, thời gian huấn luyện tăng vọt từ $178.34$ giây lên $370.93$ giây (tăng hơn $200\%$) và bộ nhớ RAM tăng từ $4.6\text{GB}$ lên $8.5\text{GB}$ do độ phức tạp của Self-Attention tỷ lệ thuận với $O(T^2)$.
  - Trong khi đó, phần lớn cảm xúc cốt lõi của người viết đánh giá phim thường nằm ở câu mở đầu hoặc câu tóm tắt ở cuối. Nếu việc tăng thêm chiều dài chuỗi chỉ mang lại sự cải thiện rất nhỏ hoặc bão hòa về Accuracy, việc tiếp tục tăng sẽ gây lãng phí nghiêm trọng tài nguyên tính toán và bộ nhớ GPU mà không đem lại giá trị thực tiễn.

---

## 5. Nguyên tắc Phương pháp luận: Phân biệt Quan sát và Suy diễn

Trong nghiên cứu khoa học NLP, cần tuân thủ nghiêm ngặt chuỗi lập luận:
$$\text{Observation (Quan sát số liệu)} \longrightarrow \text{Evidence (Bằng chứng thực nghiệm)} \longrightarrow \text{Interpretation (Suy diễn khoa học)}$$

* **Không được nhảy trực tiếp:**
  $$\text{Accuracy} = 86.0\% \centernot\implies \text{Mô hình đã hoàn toàn hiểu sâu sắc cảm xúc con người.}$$
* **Lập luận khoa học đúng đắn:**
  1. *Observation:* Mô hình đạt Accuracy $86.0\%$ trên IMDb test set và F1 $0.8649$.
  2. *Evidence:* Mô hình phân loại đúng phần lớn các câu có từ khóa cảm xúc rõ ràng; nhưng nhầm lẫn nghiêm trọng ở các câu trào phúng (sarcasm) và các câu bị cắt cụt do `max_length=128`.
  3. *Interpretation:* Mô hình BERT đã học được cách bắt cặp các mẫu từ ngữ bề mặt và tương quan ngữ cảnh cục bộ tốt hơn mô hình ngẫu nhiên, nhưng khả năng lý luận ngữ dụng (pragmatics) và xử lý ngữ cảnh siêu dài vẫn còn những giới hạn cơ bản.
