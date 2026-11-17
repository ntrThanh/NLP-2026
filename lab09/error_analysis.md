# Lab 09 — Error Analysis & Context Analysis

## 1. Qualitative Error Analysis (Experiment 6)

Dưới đây là bảng phân tích 15 mẫu lỗi tiêu biểu của mô hình BERT Token Classification trên tập kiểm thử CoNLL-2003 (gồm 5 False Positives, 5 False Negatives, và 5 Boundary & Type Errors).

| # | Text / Context | Gold | Prediction | Error Type | Explanation |
|---|---|---|---|---|---|
| **FP1** | `SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .` | `O` | `LOC: LUCKY` | Spurious Entity | Tiêu đề viết hoa toàn bộ (All-caps). Từ "LUCKY" nằm kẹp giữa các thực thể địa danh/quốc gia "JAPAN" và "CHINA", khiến mô hình bị đánh lừa về mặt ngữ cảnh và chữ hoa. |
| **FP2** | `...with a lucky 2-1 win against Syria in a Group C championship match...` | `O` | `MISC: C` | Spurious Entity | "Group C" chỉ bảng đấu giải thể thao. Quy ước gán nhãn CoNLL coi chữ cái bảng đấu là O, nhưng mô hình lại nhận diện "C" là một thực thể MISC độc lập. |
| **FP3** | `GOLF - ZIMBABWE OPEN SECOND ROUND SCORES .` | `O` | `LOC: GOLF` | Spurious Entity | Tên môn thể thao "GOLF" ở đầu bản tin viết hoa toàn bộ, mô hình nhầm lẫn vị trí mở đầu bản tin với tên địa danh xuất bản (dateline). |
| **FP4** | `CRICKET - LARA SUFFERS MORE AUSTRALIAN TOUR MISERY .` | `O` | `MISC: AUSTRALIAN TOUR` | Spurious Entity | "AUSTRALIAN TOUR" là cụm danh từ chỉ chuyến du đấu. Mô hình gán nhãn cả cụm là sự kiện thể thao MISC thay vì chỉ coi đây là danh từ chung. |
| **FP5** | `NFL AMERICAN FOOTBALL-COLTS CLOBBER EAGLES TO STAY IN PLAYOFF HUNT .` | `O` | `PER: CLOBBER` | Spurious Entity | Động từ "CLOBBER" (đè bẹp) viết hoa toàn bộ nằm giữa hai tên đội bóng "COLTS" và "EAGLES", khiến mô hình nhầm là tên một vận động viên (PER). |
| **FN1** | `RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .` | `LOC: ITALY` | `O` | Missing Entity | Headline viết hoa toàn bộ làm mất tín hiệu hình thái viết hoa chữ cái đầu (casing priors), dẫn đến việc bỏ sót thực thể quốc gia quen thuộc. |
| **FN2** | `RUGBY UNION - LITTLE TO MISS CAMPESE FAREWELL .` | `PER: CAMPESE` | `O` | Missing Entity | Tên riêng David Campese xuất hiện ở dạng họ "CAMPESE" trong tiêu đề hoa toàn bộ; mô hình không nhận diện được danh từ riêng hiếm trong ngữ cảnh này. |
| **FN3** | `CRICKET - AUSTRALIA V WEST INDIES WORLD SERIES SCOREBOARD .` | `LOC: WEST INDIES` | `O` | Missing Entity | Thực thể đa từ "WEST INDIES" trong bảng điểm vắn tắt thiếu ngữ cảnh cú pháp câu hoàn chỉnh (thiếu động từ/giới từ), dẫn tới bỏ sót cả cụm. |
| **FN4** | `CRICKET - AUSTRALIA V WEST INDIES WORLD SERIES SCOREBOARD .` | `MISC: WORLD SERIES` | `O` | Missing Entity | "WORLD SERIES" là tên giải đấu (MISC), nhưng bị bỏ qua do cấu trúc tiêu đề nén và tokenization dạng all-caps. |
| **FN5** | `League teams after games played on Thursday ( tabulate under...` | `ORG: League` | `O` | Missing Entity | CoNLL-2003 gán nhãn "League" là ORG trong chú thích bảng, nhưng mô hình coi đây là danh từ chung bình thường (O). |
| **B1** | `...after the 1995 World Cup , where he took issue...` | `MISC: 1995 World Cup` | `MISC: World Cup` | Wrong Boundary | Lỗi ranh giới: Nhãn chuẩn CoNLL bao gồm cả năm "1995" (`B-MISC: 1995`, `I-MISC: World`, `I-MISC: Cup`). Mô hình gán "1995" là O và chỉ nhận diện "World Cup". |
| **B2** | `SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .` | `LOC: JAPAN` | `PER/ORG: JAPAN` | Wrong Entity Type | Hoán dụ thể thao (metonymy): Trong tin tức thể thao, tên quốc gia đại diện cho đội tuyển thi đấu. Mô hình dự đoán nhãn ORG/PER thay vì LOC theo quy ước CoNLL. |
| **B3** | `Defender Hassan Abbas rose to intercept... top corner of Bitar 's goal .` | `PER: Bitar` | `ORG: Bitar` | Wrong Entity Type | Họ "Bitar" đi kèm sở hữu cách trước danh từ "goal" khiến mô hình nhầm lẫn thủ môn với tên câu lạc bộ/tổ chức sở hữu khung thành. |
| **B4** | `RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .` | `PER: CUTTITTA` | `ORG: CUTTITTA` | Wrong Entity Type | Họ vận động viên Cuttitta xuất hiện đơn lẻ trong tiêu đề thể thao hoa toàn bộ, mô hình nhầm thành tên đội thể thao (ORG). |
| **B5** | `SOCCER - LATE GOALS GIVE JAPAN WIN OVER SYRIA .` | `LOC: JAPAN` | `ORG: JAPAN` | Wrong Entity Type | Tương tự B2, ngữ cảnh động từ "GIVE JAPAN WIN" kích hoạt biểu diễn đội bóng (ORG), nhưng nhãn gốc quy định là địa danh quốc gia (LOC). |

---

## 2. Context Analysis (Đa nghĩa theo ngữ cảnh)

Bài toán NER phụ thuộc sâu sắc vào ngữ cảnh xuất hiện của từ. Trong tập CoNLL-2003 Test, cùng một surface form có thể mang các nhãn thực thể hoàn toàn khác nhau:

### Ví dụ 1: `Barcelona`
- **Ngữ cảnh 1 (LOC):**
  > *"FIFA 's players ' status committee , meeting in Barcelona , decided that although the Udinese document was basically valid..."*
  - **Nhãn:** `LOC` (Địa điểm/Thành phố).
  - **Dấu hiệu:** Đi sau giới từ chỉ vị trí *"meeting in"*, biểu thị địa điểm diễn ra cuộc họp.
- **Ngữ cảnh 2 (ORG):**
  > *"Defenders - Paulinho Santos ( Porto ) , ... Fernando Couto ( Barcelona ) , Helder Cristovao ( Deportivo Coruna ) ..."*
  - **Nhãn:** `ORG` (Câu lạc bộ bóng đá).
  - **Dấu hiệu:** Nằm trong dấu ngoặc đơn sau tên cầu thủ, song song với các câu lạc bộ bóng đá khác (Porto, Sporting, Juventus).

### Ví dụ 2: `Leicester`
- **Ngữ cảnh 1 (LOC):**
  > *"Barbarians - ... Darren Garforth ( Leicester ) , 2 - Norm Hewitt ( New Zealand ) ..."*
  - **Nhãn:** `LOC` (Địa phương xuất thân của cầu thủ).
- **Ngữ cảnh 2 (ORG):**
  > *"Leicester 1 ( Marshall 78 ) Blackburn 1 ( Sutton 34 ) ."*
  - **Nhãn:** `ORG` (Đội bóng tham gia trận đấu).
  - **Dấu hiệu:** Đứng đầu dòng tỷ số trận đấu, theo sau là bàn thắng và tên cầu thủ ghi bàn.

### Ví dụ 3: `Tasmania`
- **Ngữ cảnh 1 (LOC):**
  > *"Score on the first day of the four-day Sheffield Shield match between Tasmania and Victoria at Bellerive Oval on Friday :"*
  - **Nhãn:** `LOC` (Vùng lãnh thổ/bang).
- **Ngữ cảnh 2 (ORG):**
  > *"Tasmania 352 for three ( David Boon 106 not out , Shaun Young 86 not out ) v Victoria ."*
  - **Nhãn:** `ORG` (Đội cricket bang Tasmania).
  - **Dấu hiệu:** Chủ ngữ thực hiện hành động ghi 352 runs, đối đầu với đội Victoria.

---

## 3. Lợi thế của Contextual Representation so với Static Representation

Trong các phương pháp biểu diễn từ tĩnh (Static Word Embeddings như Word2Vec, GloVe - đã học ở Lab 03):
- Mỗi từ vựng chỉ được ánh xạ vào một vector duy nhất: $\mathbf{v}_{\text{Barcelona}} \in \mathbb{R}^d$.
- Do vector là bất biến bất kể câu chứa nó là gì, mô hình baseline buộc phải dựa vào các đặc trưng thủ công xung quanh (từ đứng trước/sau, cờ viết hoa) hoặc chọn nhãn chiếm đa số trong tập huấn luyện. Khi từ xuất hiện ở nghĩa thiểu số (ví dụ đội bóng thay vì thành phố), biểu diễn tĩnh hoàn toàn bất lực.

Ngược lại, với kiến trúc Pretrained Transformer (BERT):
- Biểu diễn của token được tính toán động thông qua cơ chế Multi-Head Self-Attention:
  $$\mathbf{h}_i = \text{Transformer}(\mathbf{x}_1, \dots, \mathbf{x}_n)_i$$
- Khi token `Barcelona` đi kèm cụm *"meeting in ..."*, các attention heads tập trung trọng số vào giới từ *"in"* và danh từ *"meeting"*, làm vector ẩn dịch chuyển về vùng không gian của thực thể địa lý (`LOC`).
- Khi token `Barcelona` xuất hiện bên cạnh danh sách cầu thủ hoặc tỷ số trận đấu (*"1 - 0"*, *"Blackburn"*), self-attention hướng về các token thể thao, kéo vector ẩn về không gian biểu diễn tổ chức/câu lạc bộ (`ORG`).
- Nhờ đó, Transformer giải quyết triệt để tính đa nghĩa ngữ cảnh (polysemy / metonymy) mà không cần xây dựng từ điển hay luật gán nhãn thủ công.
