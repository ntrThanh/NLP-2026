"""Build tokenizer_alignment.ipynb and qa_finetuning.ipynb (executed afterwards with nbconvert on server22).

    python build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb qa_finetuning.ipynb
"""
import nbformat as nbf

META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata = META
    nb.cells = cells
    nbf.write(nb, path)
    print('wrote', path)


# ════════════════════════════════════════════════════════════════════════
# tokenizer_alignment.ipynb
# ════════════════════════════════════════════════════════════════════════
tok_cells = [
    md("""
# Lab 10 — Tokenization & Answer Span Alignment

Notebook này kiểm tra từng bước biến một mẫu SQuAD `(question, context, answer_text, answer_start)` thành nhãn
`(start_positions, end_positions)` ở mức token:

1. Cấu trúc input `[CLS] question [SEP] context [SEP]`: `input_ids`, `attention_mask`, `token_type_ids`, `sequence_ids`
2. Character index ↔ word index ↔ token index (Calculation Exercise 3)
3. `offset_mapping` và thuật toán căn chỉnh character span → token span
4. Các trường hợp biên: answer bắt đầu/kết thúc giữa token, answer bị truncation, answer nằm ngoài window
5. Sliding window: `stride`, `return_overflowing_tokens`, `overflow_to_sample_mapping`
6. Thống kê độ dài context/answer của SQuAD và kiểm tra hàm EM/F1
"""),
    code("""
import collections
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from qa_model import char_to_token_span, prepare_train_features, squad_em, squad_f1, normalize_answer

pd.set_option('display.width', 200, 'display.max_columns', 30)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
squad = load_dataset('rajpurkar/squad')
print(squad)
"""),
    md("## 1. Cấu trúc input của QA"),
    code("""
question = 'What is the capital of France?'
context = 'The capital of France is Paris.'
enc = tokenizer(question, context, return_offsets_mapping=True)

df = pd.DataFrame({
    'token': tokenizer.convert_ids_to_tokens(enc['input_ids']),
    'input_id': enc['input_ids'],
    'attention_mask': enc['attention_mask'],
    'token_type_id': enc['token_type_ids'],
    'sequence_id': enc.sequence_ids(),
    'offset': enc['offset_mapping'],
})
df.T
"""),
    md("""
**Nhận xét:**
- `input_ids`: id trong vocab của `bert-base-cased` (101 = `[CLS]`, 102 = `[SEP]`).
- `attention_mask`: 1 cho token thật, 0 cho padding (ở đây chưa padding nên toàn 1).
- `token_type_ids`: 0 cho segment A (`[CLS]` + question + `[SEP]` đầu), 1 cho segment B (context + `[SEP]` cuối).
- `sequence_ids()`: `None` cho special tokens, 0 = question, 1 = context. **Đây** mới là thứ ta dùng để biết token nào thuộc context — không dựa vào `token_type_ids`.
- `offset_mapping`: vị trí ký tự `(start, end)` **trong chuỗi gốc tương ứng** (question cho phần 0, context cho phần 1); special tokens có `(0, 0)`.

Không phải Transformer nào cũng có `token_type_ids`: RoBERTa/DistilBERT không dùng segment embedding.
"""),
    code("""
for name in ['bert-base-cased', 'roberta-base', 'distilbert-base-cased']:
    t = AutoTokenizer.from_pretrained(name)
    e = t(question, context)
    print(f"{name:24s} keys={list(e.keys())}")
    print(' ' * 24, t.convert_ids_to_tokens(e['input_ids']))
"""),
    md("""
RoBERTa dùng `<s> question </s></s> context </s>` và **không** trả về `token_type_ids`. Tokenizer của DistilBERT vẫn sinh `token_type_ids` (do dùng chung WordPiece của BERT) nhưng `DistilBertModel` **không có segment embedding** và không nhận tham số này — tức là không thể suy ra việc model có dùng `token_type_ids` chỉ từ output của tokenizer.
Vì vậy code trong `qa_model.py` chỉ đưa `token_type_ids` vào model nếu tokenizer có sinh ra (`MODEL_INPUTS` được lọc theo cột có thật),
và xác định phần context bằng `sequence_ids()` (hoạt động cho mọi tokenizer fast).
"""),
    md("## 2. Calculation Exercise 3 — character index vs word index vs token index"),
    code("""
answer = 'Paris'
answer_start = context.index(answer)
answer_end = answer_start + len(answer)
print('characters :', list(enumerate(context)))
print(f'answer_start (first char) = {answer_start}, last char = {answer_end - 1}, exclusive end = {answer_end}')
print('context[answer_start:answer_end] =', repr(context[answer_start:answer_end]))

words = context.split()
print('word index of Paris (whitespace split):', [i for i, w in enumerate(words) if 'Paris' in w], '->', words)

s, e, status = char_to_token_span(enc['offset_mapping'], enc.sequence_ids(), answer_start, answer_end)
toks = tokenizer.convert_ids_to_tokens(enc['input_ids'])
print(f'start_token={s} ({toks[s]}), end_token={e} ({toks[e]}), status={status}')
print('decoded:', tokenizer.decode(enc['input_ids'][s:e + 1]))
"""),
    code("""
# Đổi answer thành "capital of France" (Calculation Exercise 1b)
a = 'capital of France'
a_s = context.index(a); a_e = a_s + len(a)
s, e, _ = char_to_token_span(enc['offset_mapping'], enc.sequence_ids(), a_s, a_e)
print(f'char span=[{a_s},{a_e}) -> token span=({s},{e}) = {toks[s:e+1]}')
print('Trong riêng context (bỏ [CLS] + 7 token question + [SEP] = 9 token):', (s - 9, e - 9))
"""),
    md("""
Ba loại chỉ số của cùng đáp án "Paris": **character** 25→30, **word** 5 (`'Paris.'` dính dấu chấm), **token** 14.
Offset mapping là cầu nối duy nhất đáng tin cậy giữa ký tự và token.

## 3. Thuật toán căn chỉnh trên dữ liệu SQuAD thật

Hàm `char_to_token_span` (trong `qa_model.py`) cài đặt đúng pseudo-code của đề:

```python
for token_idx, (start, end) in enumerate(offsets):       # chỉ duyệt token thuộc context
    if start <= answer_start < end: start_token = token_idx
    if start < answer_end <= end:  end_token = token_idx
```
và trả về `(0, 0)` (trỏ vào `[CLS]`) nếu answer không nằm trọn trong window.
"""),
    code("""
ex = squad['train'][0]
a_text, a_start = ex['answers']['text'][0], ex['answers']['answer_start'][0]
a_end = a_start + len(a_text)
enc = tokenizer(ex['question'], ex['context'], return_offsets_mapping=True, truncation='only_second', max_length=384)
s, e, status = char_to_token_span(enc['offset_mapping'], enc.sequence_ids(), a_start, a_end)
toks = tokenizer.convert_ids_to_tokens(enc['input_ids'])
print('Q:', ex['question']); print('Gold:', a_text, '| chars', (a_start, a_end))
print('token span:', (s, e), toks[s:e + 1])
print('offsets of span:', enc['offset_mapping'][s:e + 1])
recovered = ex['context'][enc['offset_mapping'][s][0]:enc['offset_mapping'][e][1]]
print('recovered from offsets:', repr(recovered), '| match =', recovered == a_text)
"""),
    md("### Kiểm tra toàn bộ tập train: answer khôi phục từ token span có khớp gold không?"),
    code("""
def audit(split, max_length=384, stride=128, n=None):
    data = squad[split] if n is None else squad[split].select(range(n))
    stats = collections.Counter(); mid_examples = []
    for ex in data:
        a_text, a_start = ex['answers']['text'][0], ex['answers']['answer_start'][0]
        a_end = a_start + len(a_text)
        enc = tokenizer(ex['question'].lstrip(), ex['context'], truncation='only_second', max_length=max_length,
                        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True)
        found = False
        for w in range(len(enc['input_ids'])):
            offs, seq = enc['offset_mapping'][w], enc.sequence_ids(w)
            s, e, status = char_to_token_span(offs, seq, a_start, a_end)
            stats['windows'] += 1
            stats['window_' + status] += 1
            if status == 'ok' and not found:
                found = True
                rec = ex['context'][offs[s][0]:offs[e][1]]
                stats['exact_recover'] += rec == a_text
                stats['normalized_recover'] += normalize_answer(rec) == normalize_answer(a_text)
                mid_s, mid_e = offs[s][0] != a_start, offs[e][1] != a_end
                stats['start_mid_token'] += mid_s; stats['end_mid_token'] += mid_e
                if (mid_s or mid_e) and len(mid_examples) < 8:
                    mid_examples.append({'gold': a_text, 'recovered': rec,
                                         'tokens': tokenizer.convert_ids_to_tokens(enc['input_ids'][w][s:e + 1]),
                                         'start_mid': mid_s, 'end_mid': mid_e})
        stats['examples'] += 1
        stats['answer_found_in_some_window'] += found
    return stats, mid_examples

stats, mid_examples = audit('train')
total = stats['examples']
print({k: v for k, v in stats.items()})
for k in ['answer_found_in_some_window', 'exact_recover', 'start_mid_token', 'end_mid_token']:
    print(f"{k:30s}: {stats[k]:6d} / {total} = {100 * stats[k] / total:.3f}%")
pd.DataFrame(mid_examples)
"""),
    md("""
**Trường hợp answer bắt đầu/kết thúc giữa token.** Một số ít gold answer cắt ngang một token (ví dụ gold `"1"` trong token `"1st"`,
hoặc gold không lấy hậu tố `##s`). Theo quy tắc `start <= answer_start < end`, token chứa ký tự biên được chọn, nên khi giải mã lại
bằng offset ta thu được **cả token** → span dài hơn gold một chút. Đây là giới hạn cứng của mọi model span-prediction trên subword:
những mẫu này không bao giờ đạt EM bằng cách giải mã thông thường (nhưng vẫn được F1 một phần). Tỷ lệ rất nhỏ (xem số liệu trên).

## 4. Truncation & sliding window
"""),
    code("""
long_ex = max(squad['validation'].select(range(2000)), key=lambda x: len(x['context']))
print('context chars:', len(long_ex['context']), '| context tokens:',
      len(tokenizer(long_ex['context'])['input_ids']))
print('Q:', long_ex['question']); print('Gold:', long_ex['answers'])

inputs = tokenizer(long_ex['question'], long_ex['context'], truncation='only_second', max_length=384, stride=128,
                   return_overflowing_tokens=True, return_offsets_mapping=True)
print('keys:', list(inputs.keys()))
print('number of windows:', len(inputs['input_ids']))
print('overflow_to_sample_mapping:', inputs['overflow_to_sample_mapping'])
a_start = long_ex['answers']['answer_start'][0]; a_end = a_start + len(long_ex['answers']['text'][0])
rows = []
for w in range(len(inputs['input_ids'])):
    offs, seq = inputs['offset_mapping'][w], inputs.sequence_ids(w)
    ctx = [o for o, s_ in zip(offs, seq) if s_ == 1]
    s, e, status = char_to_token_span(offs, seq, a_start, a_end)
    rows.append({'window': w, 'n_tokens': len(inputs['input_ids'][w]), 'ctx_char_range': (ctx[0][0], ctx[-1][1]),
                 'n_ctx_tokens': len(ctx), 'label': (s, e), 'status': status,
                 'decoded': tokenizer.decode(inputs['input_ids'][w][s:e + 1]) if status == 'ok' else '[CLS]'})
pd.DataFrame(rows)
"""),
    code("""
# Overlap giữa 2 window liên tiếp = stride token (tính trên phần context)
w0 = [o for o, s_ in zip(inputs['offset_mapping'][0], inputs.sequence_ids(0)) if s_ == 1]
w1 = [o for o, s_ in zip(inputs['offset_mapping'][1], inputs.sequence_ids(1)) if s_ == 1]
overlap = [o for o in w1 if o in set(w0)]
print('context tokens in window 0:', len(w0), '| window 1:', len(w1), '| shared tokens:', len(overlap))
print('window 1 starts at char', w1[0][0], '= window 0 char', w0[-len(overlap)][0])
"""),
    md("""
**Giải thích các trường trả về:**
- `input_ids` / `attention_mask` / `token_type_ids`: mỗi window là một feature riêng, question được lặp lại ở mọi window.
- `offset_mapping`: offset **luôn tính theo context gốc**, nên ta biết window nào phủ đoạn ký tự nào.
- `overflow_to_sample_mapping`: feature *i* thuộc example nào — cần để gom các window về lại một câu hỏi khi đánh giá.
- `stride=128`: số token context **chồng lấn** giữa hai window liên tiếp, đảm bảo một answer nằm gần biên window vẫn xuất hiện
  trọn vẹn trong ít nhất một window.
- Window không chứa trọn answer được gán nhãn `(0, 0)` = `[CLS]` ("không có đáp án trong window này").

### Thu nhỏ `max_length` để thấy rõ ảnh hưởng của truncation
"""),
    code("""
sub = squad['validation'].select(range(2000))
rows = []
for L in [128, 256, 384]:
    feats = sub.map(prepare_train_features, batched=True, remove_columns=sub.column_names,
                    fn_kwargs=dict(tokenizer=tokenizer, max_length=L, stride=L // 3))
    inw = np.array(feats['answer_in_window'])
    rows.append({'max_length': L, 'stride': L // 3, 'features': len(feats),
                 'windows/example': round(len(feats) / len(sub), 3),
                 '% windows containing answer': round(100 * inw.mean(), 2)})
pd.DataFrame(rows)
"""),
    md("""
Với `max_length=128` mỗi câu hỏi bị chia thành nhiều window hơn, phần lớn window **không** chứa đáp án (nhãn `[CLS]`).
Nếu chỉ `truncation` mà không dùng `return_overflowing_tokens`, các answer nằm sau token thứ ~110 của context sẽ **mất hẳn**.

## 5. Thống kê độ dài trong SQuAD
"""),
    code("""
val = squad['validation']
ctx_len = np.array([len(x) for x in tokenizer(list(val['context']))['input_ids']])
q_len = np.array([len(x) for x in tokenizer(list(val['question']))['input_ids']])
ans_words = np.array([len(a['text'][0].split()) for a in val['answers']])
print(f'context tokens: mean={ctx_len.mean():.1f} median={np.median(ctx_len):.0f} max={ctx_len.max()}')
print(f'question tokens: mean={q_len.mean():.1f} max={q_len.max()}')
for L in [128, 256, 384]:
    print(f'  % questions whose question+context > {L}: {100 * ((ctx_len + q_len) > L).mean():.1f}%')
print('answer length (words) buckets:',
      collections.Counter(np.select([ans_words == 1, ans_words <= 3, ans_words <= 6], ['1', '2-3', '4-6'], '7+')))
"""),
    md("## 6. Kiểm tra hàm metric với các ví dụ tính tay (calculations.md)"),
    code("""
cases = [('France', 'Paris France'), ('the city of Paris', 'Paris'), ('New York', 'New York City'),
         ('Paris.', 'the Paris'), ('Paris', 'Paris')]
pd.DataFrame([{'prediction': p, 'gold': g, 'norm_pred': normalize_answer(p), 'norm_gold': normalize_answer(g),
               'EM': squad_em(p, g), 'F1': round(squad_f1(p, g), 3)} for p, g in cases])
"""),
    md("Kết quả khớp với phần tự tính trong `calculations.md` (0.667, 0.500, 0.800, EM=1 sau chuẩn hóa)."),
]

# ════════════════════════════════════════════════════════════════════════
# qa_finetuning.ipynb
# ════════════════════════════════════════════════════════════════════════
ft_cells = [
    md("""
# Lab 10 — Fine-tuning `bert-base-cased` cho Extractive QA (SQuAD v1.1)

Huấn luyện được chạy bằng script (để chạy nền trên server22, tránh timeout của notebook):

```bash
python qa_model.py --tag main --max_length 384 --stride 128 --epochs 2 --save_model          # Experiment 2
python qa_model.py --tag len128 --max_length 128 --stride 42  --epochs 1 --train_samples 20000  # Experiment 1
python qa_model.py --tag len256 --max_length 256 --stride 85  --epochs 1 --train_samples 20000
python qa_model.py --tag len384 --max_length 384 --stride 128 --epochs 1 --train_samples 20000
python baseline_qa.py
python analyze_qa.py
```

Notebook này nạp lại log/checkpoint, trình bày learning curves, kết quả và chạy suy luận với model đã fine-tune.
Phần thân của vòng lặp huấn luyện (loss tường minh) được trích ở mục 1.
"""),
    code("""
import json, inspect
import pandas as pd
from IPython.display import Image, display
import qa_model
from qa_model import qa_loss, load_qa_model, answer_question
pd.set_option('display.max_colwidth', 120)
print(inspect.getsource(qa_loss))
"""),
    md("## 1. Cấu hình & thống kê feature của model chính"),
    code("""
summary = json.load(open('outputs/main/summary.json'))
cfg = summary['config']
pd.Series({k: summary[k] for k in ['train_examples', 'train_features', 'windows_per_example',
                                   'features_answer_in_window_pct', 'dev_features', 'test_features',
                                   'train_time_sec', 'gpu']} | {f'cfg.{k}': v for k, v in cfg.items()})
"""),
    md("## 2. Learning curves (training loss, validation loss, EM, F1)"),
    code("""
hist = pd.DataFrame(json.load(open('outputs/main/history.json')))
display(hist.round(4))
display(Image('learning_curves.png'))
"""),
    md("## 3. Kết quả trên test (SQuAD v1.1 dev chính thức, 10,570 câu hỏi)"),
    code("""
display(pd.read_csv('results.csv'))
display(Image('em_f1_analysis.png'))
"""),
    md("## 4. Experiment 1 — Context length"),
    code("""
rows = []
for L in [128, 256, 384]:
    s = json.load(open(f'outputs/len{L}/summary.json'))
    rows.append({'max_length': L, 'stride': s['config']['stride'], 'windows/example': round(s['windows_per_example'], 3),
                 '% train windows w/ answer': round(s['features_answer_in_window_pct'], 2),
                 'EM': round(s['test']['exact_match'], 2), 'F1': round(s['test']['f1'], 2),
                 'train_time_min': round(s['train_time_sec'] / 60, 1)})
pd.DataFrame(rows)
"""),
    md("## 5. Answer length & lexical overlap"),
    code("""
analysis = json.load(open('outputs/analysis.json'))
display(pd.DataFrame(analysis['answer_length']))
display(Image('answer_length_analysis.png'))
display(pd.DataFrame(analysis['lexical_overlap']))
display(pd.DataFrame(analysis['robustness']))
"""),
    md("## 6. Suy luận với model đã fine-tune"),
    code("""
tok, model, device = load_qa_model('outputs/main/model')
ctx = 'The capital of France is Paris. The French government is based in Paris.'
for q in ['What city is the capital of France?', 'Where is the French government based?',
          'What is the capital of Germany?']:
    r = answer_question(q, ctx, tok, model, device)
    print(f"{q:45s} -> {r['text']!r:12s} chars=({r['start_char']},{r['end_char']}) score={r['score']:.2f}")
"""),
    md("""
Câu hỏi cuối ("capital of Germany") **không có đáp án** trong context nhưng model SQuAD v1.1 vẫn buộc phải trả về một span
→ minh họa hạn chế của extractive QA không có nhánh no-answer (Mục 26).
"""),
]

if __name__ == '__main__':
    save(tok_cells, 'tokenizer_alignment.ipynb')
    save(ft_cells, 'qa_finetuning.ipynb')
