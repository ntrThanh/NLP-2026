# Lab 09 — Theory & Calculation Exercises

## Calculation Exercise 1 — BIO Tagging

Cho câu:

    Elon Musk founded SpaceX in California.

Với annotation:
- Elon Musk → PER
- SpaceX → ORG
- California → LOC

BIO labels:

| Token        | BIO Label   |
|:-------------|:------------|
| Elon         | B-PER       |
| Musk         | I-PER       |
| founded      | O           |
| SpaceX       | B-ORG       |
| in           | O           |
| California   | B-LOC       |
| .            | O           |

---

## Calculation Exercise 2 — Entity Boundary

Cho cụm:

    University of California Berkeley

Toàn bộ cụm là một organization.

| Token        | BIO Label   |
|:-------------|:------------|
| University   | B-ORG       |
| of           | I-ORG       |
| California   | I-ORG       |
| Berkeley     | I-ORG       |

**Vì sao không thể gán tất cả token là ORG (không phân biệt B/I)?**

Nếu chỉ gán tất cả là `ORG` mà không phân biệt `B-ORG` / `I-ORG`, ta không thể xác định ranh giới (boundary) giữa các entity liền kề.

Ví dụ: nếu ta có câu "University of California Berkeley and Stanford University", và tất cả đều gán `ORG`, ta không thể biết đâu là entity thứ nhất kết thúc và đâu là entity thứ hai bắt đầu. Tag `B-` đánh dấu điểm bắt đầu một entity mới, giúp phân tách rõ ràng các entity kề nhau cùng loại.

---

## Calculation Exercise 3 — Shape

Cho:
- B = 8 (batch size)
- T = 128 (sequence length)
- d = 768 (hidden dimension)
- C = 9 (số NER labels: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O)

| Tensor        | Shape           | Giải thích                                       |
|:--------------|:----------------|:-------------------------------------------------|
| Hidden states | (8, 128, 768)   | Mỗi token trong mỗi câu có một vector 768 chiều |
| Logits        | (8, 128, 9)     | Mỗi token được dự đoán thuộc 1 trong 9 lớp      |
| Labels        | (8, 128)        | Mỗi token có đúng 1 nhãn ground-truth            |

Phép tính classification head:

    Z = H @ W + b

với W có shape (768, 9) và b có shape (9,).
