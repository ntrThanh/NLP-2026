# Reflection & Learning Check — Lab 01

---

## 16. Reflection

### 1. Prediction nào của em sai hoặc có độ lệch?

Ở phần dự đoán kích thước từ điển ban đầu, em dự đoán tập từ vựng thô khoảng 300k từ (do được thực nghiệm từ trước). Khi chạy thực tế trên dữ liệu thô chưa lowercasing, từ điển lên tới 240.792 từ. Sau khi chuyển chữ thường và lọc ký tự cơ bản ở Pipeline A thì từ điển mới rút về 186.582 từ, tương đối sát với dự đoán.

Dự đoán về độ thưa ma trận đạt với kết quả thực tế đạt độ thưa 99.93%, chỉ có 0.07% phần tử khác 0. Dự đoán về việc tìm kiếm từ khóa không bảo đảm đúng nghĩa cũng được kiểm chứng rõ ràng qua hiện tượng lệch từ vựng.

### 2. Kết quả nào làm em bất ngờ nhất?

Việc loại bỏ stopwords tiếng Anh ở Pipeline B chỉ làm giảm kích thước từ điển từ 193.837 xuống 192.764 từ, tức là chỉ giảm hơn 1.000 từ, khoảng 0.5% từ vựng.

Số từ dừng này chiếm tới 1/3 tổng số token xuất hiện trong toàn bộ văn bản. Loại bỏ chúng, số phần tử khác 0 giảm mạnh và chất lượng tìm kiếm tăng lên do giảm từ gây nhiễu.

### 3. Experiment nào cung cấp evidence mạnh nhất?

Thí nghiệm tiền xử lý ở phần 9 cho thấy tiền xử lý là một quyết định xây dựng mô hình.

Khi chuyển sang dùng subword tokenizer ở Pipeline C, kích thước từ điển được cố định ở 30.522 từ và loại bỏ hiện tượng từ ngoài từ điển OOV = 0%.

### 4. Failure case quan trọng nhất là gì?

 Khi tìm kiếm cụm từ heart attack treatment đối chiếu với văn bản 8343 nói về liệu pháp myocardial infarction therapy.

Hai khái niệm này hoàn toàn đồng nghĩa trong y khoa nhưng không chung nhau chữ cái nào. Điểm tương đồng cosine trả về bằng đúng 0.0000, khiến hệ thống bỏ sót tài liệu quan trọng nhất.

### 5. Nếu được xây dựng lại search engine, em sẽ thay đổi điều gì?

Đầu tiên là chuyển sang dùng thuật toán Okapi BM25 để có hàm bão hòa tần suất từ tf / (tf + k1), tránh việc văn bản lặp lại một từ quá nhiều bị đẩy lên đầu như bài chia thì động từ ở văn bản 7564.

Tiếp theo là hỗ trợ tìm kiếm theo cụm từ hoặc n-gram để giữ lại các khái niệm như natural language processing thay vì bẻ vụn thành các từ đơn lẻ.

Cuối cùng là kết hợp tìm kiếm từ khóa với biểu diễn vector ngữ nghĩa dày đặc như Sentence-BERT để hệ thống hiểu được các từ đồng nghĩa và ngữ cảnh của câu hỏi.

---

## 14. AI Usage Policy

Trong bài lab này, AI được sử dụng để hỗ trợ giải thích cú pháp thư viện ma trận thưa Scipy CSR, tối ưu tốc độ tính tích vô hướng cosine similarity trên tập dữ liệu lớn và gợi ý khung unit test trong implementation.py.

---

## 15. Learning Check

### Question 1: Tại sao TF-IDF tạo ra sparse representation?

Vector của mỗi văn bản có số chiều bằng toàn bộ kích thước từ điển, trong bài này là khoảng 186.000 chiều.

Trong khi đó, mỗi văn bản bình thường chỉ dùng từ vài chục đến vài trăm từ riêng biệt. Vì vậy hầu hết các vị trí trong vector đều mang giá trị 0, tỷ lệ phần tử mang giá trị 0 lên tới hơn 99.9%, tạo thành ma trận cực kỳ thưa.

### Question 2: Tại sao một term xuất hiện trong hầu hết documents có IDF thấp?

Theo công thức idf(t) = log(N / df(t)), khi một từ xuất hiện trong hầu hết các văn bản thì df(t) xấp xỉ N, dẫn đến tỷ số N / df(t) xấp xỉ 1 và log(1) bằng 0.

Về mặt thông tin, một từ xuất hiện tràn lan ở mọi văn bản thì không có tác dụng phân loại hay phân biệt văn bản này với văn bản khác, nên được gán trọng số rất thấp.

### Question 3: Tại sao một term có IDF cao chưa chắc có TF-IDF cao trong một document?

Công thức tính là TF-IDF = TF * IDF. Điểm IDF cao chỉ cho biết từ đó hiếm gặp trong toàn bộ tập dữ liệu.

Nếu văn bản đang xét không chứa từ này thì tần suất TF bằng 0, dẫn đến TF-IDF cũng bằng 0. Một từ chỉ có TF-IDF cao khi nó vừa hiếm trong toàn bộ corpus, vừa phải xuất hiện nhiều lần trong chính văn bản đó.

### Question 4: Tại sao cosine similarity phù hợp với document vectors?

Cosine similarity đo góc lệch hướng giữa hai vector thay vì đo khoảng cách hình học tuyệt đối. 

Nhờ có bước chia cho độ dài vector theo chuẩn L2, phép đo này triệt tiêu được sự chênh lệch về độ dài văn bản. Một bài viết dài và một bài viết ngắn cùng chủ đề vẫn có vector chỉ về cùng một hướng và cho độ tương đồng cao.

### Question 5: Tại sao preprocessing có thể thay đổi search result?

Tiền xử lý làm thay đổi trực tiếp kích thước từ điển, số lượng từ, tần suất TF và trọng số IDF. 

Chuyển chữ thường giúp gộp các dạng viết hoa viết thường của cùng một từ. Bỏ từ dừng giúp lọc bớt từ nhiễu và làm nổi bật từ khóa chính. Cách tách từ theo từ đơn hay mảnh từ cũng quyết định việc văn bản có bị gặp từ lạ OOV hay bị ghép nhầm từ hay không.

### Question 6: Một failure case của TF-IDF search mà em quan sát được là gì?

Đó là trường hợp lệch từ vựng giữa câu truy vấn heart attack treatment và văn bản 8343 chứa cụm từ myocardial infarction therapy.

Hai câu này mang cùng một ý nghĩa y học nhưng không trùng nhau từ nào. Điểm tương đồng tính ra bằng 0.0000 nên hệ thống bỏ sót hoàn toàn văn bản phù hợp.

### Question 7: Failure case đó gợi ý nhu cầu về representation nào tiếp theo?

Thất bại trên gợi ý việc cần chuyển sang biểu diễn ngữ nghĩa dày đặc dựa trên giả thuyết phân phối, nghĩa là những từ xuất hiện trong ngữ cảnh tương tự nhau thì sẽ có vector nằm gần nhau trong không gian liên tục.

Chuỗi phát triển tiếp theo của NLP là đi từ TF-IDF sang Word Embedding như Word2Vec, FastText và nâng lên Contextual Embedding như BERT để giải quyết bài toán từ đồng nghĩa và từ đa nghĩa.
