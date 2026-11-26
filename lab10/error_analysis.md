# Lab 10 — Error Analysis

Model: `bert-base-cased` fine-tuned trên SQuAD v1.1 (max_length 384, stride 128, 2 epochs).
Tập đánh giá: SQuAD v1.1 dev chính thức (10,570 câu hỏi) — **EM 81.40, F1 88.73**.
Số liệu lấy từ `outputs/analysis.json`, mẫu lỗi từ `outputs/error_samples.json` (do `analyze_qa.py` sinh ra).

---

## 1. Phân loại lỗi theo boundary (Mục 25)

Mỗi prediction sai (EM = 0) được so với **gold answer cho F1 cao nhất** (SQuAD dev có tới 3–5 gold/câu hỏi), ở mức **ký tự** trong context. Hai biên được coi là "đúng" nếu phần chênh lệch giữa chúng chỉ là khoảng trắng/dấu câu/mạo từ (bị loại khi chuẩn hóa).

| Nhóm | Số câu | % của lỗi (1,966) | Ý nghĩa |
|:--|--:|--:|:--|
| Correct (EM = 1) | 8,604 | – | |
| **Wrong span — không chồng lấn** | **823** | 41.9% | chọn sai hẳn vị trí (distractor, sai kiểu đáp án) |
| Wrong start — quá dài (start sớm hơn gold) | 316 | 16.1% | end đúng, kéo thêm từ phía trước |
| Wrong start — quá ngắn (start muộn hơn gold) | 237 | 12.1% | end đúng, bỏ mất từ đầu |
| Wrong end — quá dài (end muộn hơn gold) | 287 | 14.6% | start đúng, kéo thêm phần đuôi |
| Wrong end — quá ngắn (end sớm hơn gold) | 212 | 10.8% | start đúng, cắt cụt |
| Cả hai biên sai — bao trùm gold | 46 | 2.3% | over-extended |
| Cả hai biên sai — nằm trong gold | 22 | 1.1% | partial answer |
| Cả hai biên sai — lệch chồng lấn | 23 | 1.2% | shifted |

### Thống kê #wrong start và #wrong end

| Chỉ số | Số lượng |
|:--|--:|
| Tổng số prediction sai (EM = 0) | 1,966 |
| **#wrong start** (start ≠ gold, tính cả no-overlap) | **1,467** |
| **#wrong end** (end ≠ gold, tính cả no-overlap) | **1,413** |
| #wrong start trong các lỗi có chồng lấn (1,143) | 644 |
| #wrong end trong các lỗi có chồng lấn | 590 |
| Chỉ sai start (end đúng) | 553 |
| Chỉ sai end (start đúng) | 499 |
| Sai cả hai nhưng còn chồng lấn | 91 |
| Độ dài TB prediction sai / gold tương ứng | 4.98 từ / 4.26 từ |

**Quan sát:**
1. **58% lỗi là lỗi boundary** (prediction chồng lấn gold), chỉ 42% là chọn sai hẳn vị trí. Nghĩa là model thường "tìm đúng chỗ" nhưng cắt sai biên.
2. Trong lỗi boundary, **chỉ một biên sai** chiếm áp đảo (553 + 499 = 1,052 / 1,143 = 92%). Điều này khớp với việc start và end được dự đoán bởi **hai phân phối độc lập**: thường một đầu "chắc chắn" (đỉnh softmax nhọn), đầu kia phân vân giữa 2–3 điểm dừng hợp lý.
3. **Wrong start (553) nhiều hơn wrong end (499) một chút** — trái với dự đoán ban đầu (Prediction 2 cho rằng end khó hơn). Ngoài ra lỗi "quá dài" nhiều hơn "quá ngắn" ở cả hai biên (316 > 237; 287 > 212) và prediction sai dài hơn gold trung bình 0.7 từ → model có xu hướng **over-extend**.

### Giả thuyết về nguyên nhân

- **Wrong start** chủ yếu nằm ở các **từ chức năng/định lượng đứng trước đáp án**: giới từ (`among cultures`, `to abolish…`, `South of the Tehachapis`), lượng từ/trạng từ (`about`, `over 5,100`, `between 1268 and 1273`), đại từ sở hữu (`their higher oxygen content`), danh từ định danh (`Windows 10` vs gold `10`). Các annotator **không nhất quán** khi nào lấy những từ này: có gold `"over 5,100"` nhưng có gold `"3,000 miles"` (bỏ `about`). Normalization của SQuAD chỉ bỏ `a/an/the`, không bỏ `about/over/between/to/their` → lỗi start bị "khuếch đại" bởi chính metric.
- **Wrong end** chủ yếu là **bổ ngữ phía sau** cụm danh từ: cụm giới từ (`American attack on Iraq` **in the 1991 Gulf War**), mệnh đề quan hệ, đơn vị/danh từ đầu (`Pacific` **coast**, `Killer` vs `Killer T cells`), hoặc liệt kê (`lab monitoring, adherence counseling` **, and assist patients…**). Điểm kết thúc của một NP có thể nằm ở nhiều chỗ hợp lý về ngữ pháp.
- Lý do start hơi nhiều hơn end: trong tiếng Anh, phần **đầu cụm** (determiners, quantifiers, prepositions) đa dạng và bị annotator xử lý tùy tiện hơn; phần cuối cụm thì thường dừng ở dấu câu (`,` `.` `(`) — một tín hiệu bề mặt mạnh mà model học rất dễ.

---

## 2. Bảng phân tích lỗi chi tiết (≥ 5 wrong start, ≥ 5 wrong end, ≥ 5 wrong span)

### 2.1. Wrong start (end đúng)

| # | Question | Gold | Prediction | Error type | Explanation |
|:-:|:--|:--|:--|:--|:--|
| 1 | Which mountain range influenced the split of the regions? | Tehachapis | South of the Tehachapis | wrong start (quá dài) / boundary | Model lấy nguyên cụm địa lý "South of the Tehachapis" vì cụm này là một thuật ngữ trong câu; câu hỏi hỏi **dãy núi** nên chỉ cần danh từ riêng. |
| 2 | What is the upper house of the Parliament of Victoria called? | Legislative Council | Legislative Assembly (the lower house) and the Legislative Council | wrong start (quá dài) / **multiple plausible spans** | Hai thực thể cùng kiểu đứng cạnh nhau; nhãn "(the upper house)" nằm **sau** đáp án nên start logit bị chia giữa "Legislative Assembly" và "Legislative Council". End đúng nhờ dấu ngoặc ngay sau. |
| 3 | What version of Windows supported the CBS sports app? | 10 | Windows 10 | wrong start (quá dài) / question ambiguity | Câu hỏi đã chứa "Windows", gold chỉ lấy "10". Prediction hợp lý về ngữ nghĩa nhưng thừa một từ → EM = 0, F1 = 0.67. |
| 4 | What factor may make a teacher's role vary? | cultures | among cultures | wrong start (quá dài) / boundary | Model kéo theo giới từ "among". Lỗi kiểu giới từ đầu cụm — normalization không bỏ giới từ. |
| 5 | Why do polar water bodies support a higher amount of life? | higher oxygen content | their higher oxygen content | wrong start (quá dài) / boundary | Thêm đại từ sở hữu "their". Một gold khác là "due to their higher oxygen content" → bản thân annotator cũng không thống nhất biên trái. |
| 6 | When did Kublai attack Xiangyang? | between 1268 and 1273 | 1268 and 1273 | wrong start (quá ngắn) / partial answer | Model bỏ "between" — với câu hỏi "When", model ưu tiên bắt đầu ở token số. F1 = 0.86. |
| 7 | What makes the Wells Fargo Center stand out? | its distinctive flared base | flared base | wrong start (quá ngắn) / partial answer | Bỏ tính từ "distinctive" — start rơi vào lõi danh từ thay vì đầu cụm. |
| 8 | What is the Rankine cycle sometimes called? | practical Carnot cycle | Carnot cycle | wrong start (quá ngắn) / boundary | "Carnot cycle" xuất hiện **hai lần** trong đoạn văn (lần sau không có "practical") → model khớp với tên chuẩn của chu trình, bỏ tính từ phân biệt. |

### 2.2. Wrong end (start đúng)

| # | Question | Gold | Prediction | Error type | Explanation |
|:-:|:--|:--|:--|:--|:--|
| 1 | What did the NIF try to unify Islamist opposition against? | American attack on Iraq | American attack on Iraq in the 1991 Gulf War | wrong end (quá dài) / boundary | Cụm giới từ thời gian "in the 1991 Gulf War" bổ nghĩa cho NP; model dừng ở dấu chấm cuối câu thay vì cuối NP. |
| 2 | Which coastline does Southern California touch? | Pacific | Pacific coast | wrong end (quá dài) / question ambiguity | "coastline" trong câu hỏi đã hàm ý "coast"; gold chỉ lấy tính từ "Pacific". Về ngữ nghĩa prediction không sai. |
| 3 | Secular Arab nationalism was blamed for both the defeat … as well as what type of stagnation? | economic | economic stagnation in the defeated countries, was blamed on the secular | wrong end (quá dài) / **long span drift** | End logit đặt vào một vị trí rất xa, vượt qua cả dấu phẩy và động từ chính — ví dụ hiếm về việc start/end độc lập tạo ra span vô nghĩa (start đúng nhưng end "trôi"). |
| 4 | What broadcast group is the largest operator of ABC stations? | Sinclair Broadcast Group | Sinclair | wrong end (quá ngắn) / tokenization-entity boundary | Câu hỏi chứa "broadcast group" nên model coi đó là phần của câu hỏi và chỉ lấy tên riêng. Gold lại lấy cả tên tổ chức đầy đủ. |
| 5 | What kind of T cells kill cells that are infected with pathogens? | Killer T cells | Killer | wrong end (quá ngắn) / question overlap | Tương tự: "T cells" đã có trong câu hỏi nên model chỉ trả về phần "mới". Đây là một chiến lược hợp lý nhưng không khớp quy ước annotate. |
| 6 | Besides drugs, what else do specialty pharmacies provide? | lab monitoring, adherence counseling, and assist patients with cost-containment strategies | lab monitoring, adherence counseling | wrong end (quá ngắn) / long answer | Đáp án là một danh sách dài 12+ từ; model dừng ở dấu phẩy đầu tiên của phần liệt kê không đồng dạng ("and assist…" là cụm động từ). |
| 7 | What new series continues the plot of the original Doctor Who? | The 2005 version (of Doctor Who) | 2005 | both wrong — inside gold / partial | Model chỉ lấy năm; "Doctor Who" xuất hiện trong câu hỏi nên model không lấy. |
| 8 | Where does the largest part of Kenya's power come from? | hydroelectric stations at dams | hydroelectric stations | wrong end (quá ngắn) / boundary | Cụm "at dams" là bổ ngữ địa điểm — ngược với ví dụ #1, ở đây model lại cắt quá sớm → cho thấy biên phải của NP + PP thực sự mơ hồ. |

### 2.3. Wrong span (không chồng lấn gold)

| # | Question | Gold | Prediction | Error type | Explanation |
|:-:|:--|:--|:--|:--|:--|
| 1 | Who created the theme used in 1986? | Dominic Glynn | Keff McCulloch | **distractor** | Đoạn văn liệt kê nhiều nhà soạn nhạc + năm. "(1986)" nằm cuối mệnh đề về Glynn, còn tên "Keff McCulloch" đứng ngay sau "(1986)." → model chọn tên **gần** con số 1986 nhất về vị trí thay vì đúng quan hệ cú pháp. |
| 2 | What are upper-level secondary schools called? | Gymnasium | Hauptschule | **distractor** / paraphrase | Context dùng "higher level secondary schools (Gymnasium)" còn câu hỏi dùng "upper-level". Model khớp nhầm với "lower secondary schools (Hauptschule)" — một cấu trúc song song, chỉ khác tính từ. |
| 3 | What did Marlee Matlin translate? | the national anthem | American Sign Language | **insufficient reasoning / ellipsis** | Câu: "Lady Gaga performed the national anthem, while … Marlee Matlin provided ASL translation". Cần hiểu "translation" (của bài quốc ca) — một suy luận ẩn. Model chọn đối tượng gần "Matlin" nhất. |
| 4 | What do these teachers NOT do? | teach by rote | attempt to find new invigoration for the course materials | **negation** | Câu hỏi có phủ định "NOT"; context "do not teach by rote **but** attempt to find…". Model chọn vế khẳng định sau "but" — Transformer fine-tune trên SQuAD xử lý phủ định kém vì rất hiếm mẫu dạng này. |
| 5 | How many lakes are there in Warsaw? | several | 13 | **answer-type bias / distractor** | "How many" → model ưu tiên chữ số. "13 natural reserves" có số và có "Warsaw", trong khi đáp án đúng là từ "several" (không phải số). |
| 6 | What was the ideal duty of a Newcomen engine? | 7 million | burning one bushel (94 pounds) of coal | **question ambiguity** | Câu hỏi dùng "duty" — đoạn văn có **định nghĩa** của duty ("delivered by burning one bushel…") và **giá trị** duty (7 million). Model chọn định nghĩa. |
| 7 | What type of physics model did Einstein fail to make? | self-consistent unification models | string theory | **distractor / coreference** | "Einstein tried and failed at **this endeavor**" — cần giải đồng tham chiếu "this endeavor" về câu trước. Model chọn "string theory" nằm ngay sau trong cùng câu. |
| 8 | What was the third city that was considered? | San Francisco | New Orleans | **ordinal reasoning** | Cần đếm thứ tự trong danh sách 3 thành phố; model chọn phần tử đầu tiên. Overlap từ vựng = 0. |

---

## 3. EM = 0 nhưng F1 > 0 (Experiment 3)

1,321 / 10,570 câu hỏi (12.5%) có EM = 0 nhưng F1 > 0 — đây chính là khoảng cách F1 − EM = 7.3 điểm.

| # | Question | Gold (gần nhất) | Prediction | F1 | Vì sao EM = 0 nhưng F1 > 0 |
|:-:|:--|:--|:--|:-:|:--|
| 1 | At what address did Goldenson secure a new headquarters for ABC? | 1330 Avenue of the Americas in Manhattan | 1330 Avenue of the Americas | 0.80 | Thiếu "in Manhattan": P = 4/4 = 1.0, R = 4/6 = 0.67 → F1 = 0.80 (sau khi bỏ "the"). |
| 2 | Approximately how many names were signed on an online petition…? | over 5,100 | 5,100 | 0.67 | Thiếu lượng từ "over": P = 1, R = 1/2 → F1 = 0.67. Chính xác về con số nhưng EM rất nghiêm ngặt. |
| 3 | What is floridean? | a form of starch | starch | 0.50 | Sau chuẩn hóa gold = [form, of, starch] (bỏ "a"): P = 1, R = 1/3 → F1 = 0.50. |
| 4 | where is the Victoria and Albert Museum located? | in the Brompton district of the Royal Borough of Kensington and Chelsea | Brompton district of the Royal Borough of Kensington and Chelsea | 0.95 | Chỉ thiếu "in" → gần như trùng hoàn toàn, nhưng EM = 0. |
| 5 | Which channels did Frank Marx think would be requisitioned…? | channels 2 through 6 | low-band VHF frequencies (corresponding to channels 2 through 6 | 0.62 | Prediction bao trùm gold và thừa 5 từ: P = 4/9, R = 4/4. Đồng thời end rơi giữa ngoặc đơn — dấu hiệu end logit dừng ngay trước ")". |
| 6 | What can result in creating a poverty trap? | gender inequality in education | gender inequality in education can result to low economic growth, and continued gender inequality in education | 0.48 | Over-extended: model gộp cả chuỗi nhân quả thành một span. |
| 7 | The amazon rainforest became a mostly inland forest around which global event? | last glacial maximum | glacial maximum | 0.80 | Thiếu tính từ "last" (wrong start). |

**Kết luận:** EM chỉ chấp nhận trùng khớp tuyệt đối sau chuẩn hóa, nên bất kỳ lệch biên nào (một giới từ, một lượng từ) đều cho 0 điểm. F1 cho **partial credit** dựa trên túi token chung → phản ánh tốt hơn việc model đã "tìm đúng vùng" của đáp án. Vì vậy EM đo **độ chính xác biên**, F1 đo **độ chính xác vị trí + mức độ chồng lấn**. Khoảng cách F1 − EM là thước đo trực tiếp của lỗi boundary.

---

## 4. Experiment 4 — Question/context lexical matching

Lexical overlap = tỷ lệ content word (bỏ stopword, stem thô) của câu hỏi xuất hiện trong **câu chứa đáp án** (đã loại bỏ chính đáp án). Chia theo tứ phân vị:

| Nhóm | n | Overlap TB | BERT EM | BERT F1 | Baseline EM | Baseline F1 |
|:--|--:|:-:|:-:|:-:|:-:|:-:|
| High overlap (≥ 0.80) | 2,847 | 0.930 | **87.43** | **93.53** | 24.69 | 40.05 |
| Middle | 4,767 | 0.609 | 82.90 | 89.93 | 16.07 | 29.76 |
| Low overlap (≤ 0.40) | 2,956 | 0.228 | **73.17** | **82.19** | 6.50 | 15.80 |
| **Chênh lệch High − Low** | | | **−14.3** | **−11.3** | −18.2 | −24.3 |

Ví dụ thủ công (`document_qa.py`):

| Context | Question | Overlap | Prediction |
|:--|:--|:--|:--|
| The capital of France is Paris. The French government is based in Paris. | What city is the capital of France? | cao | Paris ✓ (câu 1) |
| | Where is the French government based? | cao | Paris ✓ (câu 2) |
| | Which city hosts the French government? | thấp ("hosts" ≠ "based in") | Paris ✓ (câu 2) |

**Nhận xét:**
- BERT **có phụ thuộc** vào lexical overlap: mất 11.3 điểm F1 khi câu hỏi được diễn đạt khác với câu chứa đáp án — xác nhận Prediction 1. Tuy nhiên ở nhóm overlap thấp BERT vẫn đạt F1 82.2, gấp 5 lần baseline (15.8), nghĩa là model **không chỉ** khớp từ mà còn hiểu paraphrase (`hosts` ↔ `based in`, `company` ↔ `researchers at Google`).
- Baseline gần như sụp đổ ở nhóm overlap thấp (F1 15.8) — đúng bản chất heuristic dựa hoàn toàn vào trùng từ.
- Lỗi nhóm overlap thấp điển hình: "What was the third city that was considered?" (cần đếm), "What public entity of learning is often target of civil disobedience?" (cần nhận ra `entity of learning` ↔ `universities`).

---

## 5. Experiment 5 — Answer length

| Answer length (từ) | n | BERT EM | BERT F1 | F1 − EM | Baseline EM | Baseline F1 |
|:--|--:|:-:|:-:|:-:|:-:|:-:|
| 1 | 3,429 | **85.13** | 89.37 | 4.24 | 22.95 | 32.22 |
| 2–3 | 4,575 | 83.56 | **89.96** | 6.40 | 14.73 | 29.39 |
| 4–6 | 1,648 | 78.40 | 88.42 | 10.02 | 10.25 | 24.13 |
| 7+ | 918 | **62.09** | 80.81 | **18.72** | 3.38 | 19.54 |

**Đối chiếu Prediction 2:** Đúng về xu hướng — EM giảm đều và giảm mạnh ở nhóm 7+ (85.1 → 62.1, −23 điểm), trong khi F1 giảm ít hơn nhiều (89.4 → 80.8, −8.6 điểm). Khoảng F1 − EM tăng gấp 4 lần (4.2 → 18.7): answer dài **không** bị tìm sai vị trí nhiều hơn mà chủ yếu bị **lệch biên**. Mức EM 62% ở nhóm 7+ cao hơn khoảng 40–55% tôi dự đoán — model xử lý span dài tốt hơn kỳ vọng nhờ dấu câu giúp xác định biên. F1 của nhóm 2–3 nhỉnh hơn nhóm 1 vì với answer 1 từ, mọi lỗi đều cho F1 = 0 (không có partial credit).

---

## 6. Phân loại theo nguyên nhân (tổng hợp các nhóm lỗi Mục 24)

| Error type | Ví dụ tiêu biểu | Cơ chế |
|:--|:--|:--|
| wrong start | `among cultures`, `South of the Tehachapis` | start logit chia cho giới từ/lượng từ đứng trước |
| wrong end | `American attack on Iraq in the 1991 Gulf War` | bổ ngữ sau NP, dừng ở dấu câu |
| wrong boundary (cả hai) | `when resources (such as time or space) are bounded` | span bao trùm cả mệnh đề |
| partial answer | `1268 and 1273`, `flared base` | chỉ lấy lõi danh từ/số |
| distractor | `Keff McCulloch`, `Hauptschule` | thực thể cùng kiểu, gần vị trí từ khóa |
| insufficient context / reasoning | `American Sign Language`, `string theory` | cần ellipsis, đồng tham chiếu |
| multiple plausible spans | `Legislative Assembly … and the Legislative Council` | 2 thực thể cùng kiểu cạnh nhau |
| question ambiguity | `Windows 10` vs `10`, `duty` | câu hỏi cho phép nhiều cách hiểu |
| negation | `teach by rote` vs vế sau "but" | phủ định hiếm trong dữ liệu huấn luyện |
| tokenization | gold `"7"` bên trong token `1817` (`tokenizer_alignment.ipynb`) | 0.25% mẫu train có biên rơi giữa token → không bao giờ đạt EM |
| truncation / long context | — | với max_length 384 + stride 128, 100% answer train nằm trọn trong ít nhất một window; lỗi do truncation chỉ đáng kể ở max_length 128 (Experiment 1) |
