"""Build and execute tokenizer_alignment.ipynb and ner_finetuning.ipynb."""

import json
import os
import nbformat as nbf


def make_clean_cell(cell_type, source):
    if cell_type == "markdown":
        c = nbf.v4.new_markdown_cell(source)
    else:
        c = nbf.v4.new_code_cell(source)
    c.metadata = {}
    return c


def create_tokenizer_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    }

    cells = [
        make_clean_cell("markdown", "# Lab 09 — Tokenizer & Label Alignment\n\nPhân tích hiện tượng phân tách subwords và chiến lược căn chỉnh nhãn (label alignment) trên CoNLL-2003."),
        make_clean_cell("code", """from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset('lhoestq/conll2003')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

sample = ds['train'][0]
print("Tokens:  ", sample['tokens'])
print("NER Tags:", sample['ner_tags'])"""),
        make_clean_cell("markdown", "## 1. Subword Tokenization & Alignment Problem"),
        make_clean_cell("code", """words = ["Washington", "playing", "UNICEF", "1996-12-06", "newcomers"]
for w in words:
    subwords = tokenizer.tokenize(w)
    print(f"{w:15s} -> {subwords}")"""),
        make_clean_cell("markdown", "## 2. Alignment Strategies (Strategy A vs Strategy B)"),
        make_clean_cell("code", """tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
sample_tokens = ds['train'][0]['tokens']
sample_tags = [tag_names[t] for t in ds['train'][0]['ner_tags']]

enc = tokenizer(sample_tokens, is_split_into_words=True, return_tensors=None)
subwords = tokenizer.convert_ids_to_tokens(enc['input_ids'])
word_ids = enc.word_ids()

# Strategy A: First subword receives word tag, subsequent receive -100
strat_a_labels = []
prev = None
for wid in word_ids:
    if wid is None:
        strat_a_labels.append(-100)
    elif wid != prev:
        strat_a_labels.append(sample_tags[wid])
    else:
        strat_a_labels.append(-100)
    prev = wid

# Strategy B: Propagate tag
strat_b_labels = []
prev = None
for wid in word_ids:
    if wid is None:
        strat_b_labels.append('O')
    elif wid != prev:
        strat_b_labels.append(sample_tags[wid])
    else:
        orig = sample_tags[wid]
        strat_b_labels.append(f"I-{orig[2:]}" if orig.startswith('B-') else orig)
    prev = wid

print(f"{'Subword':<12} {'Word ID':<10} {'Strategy A':<15} {'Strategy B'}")
print("-" * 50)
for sw, wid, sa, sb in zip(subwords, word_ids, strat_a_labels, strat_b_labels):
    print(f"{sw:<12} {str(wid):<10} {str(sa):<15} {sb}")"""),
        make_clean_cell("markdown", "## 3. Loss Masking with Ignore Index (-100)"),
        make_clean_cell("code", """import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
logits = torch.randn(1, len(strat_a_labels), 9)
targets = torch.tensor([tag_names.index(l) if l != -100 else -100 for l in strat_a_labels]).unsqueeze(0)

loss = loss_fn(logits.view(-1, 9), targets.view(-1))
print("Computed loss (ignoring -100):", round(loss.item(), 4))"""),
        make_clean_cell("markdown", "## 4. Experiment 1: Tokenization Statistics"),
        make_clean_cell("code", """import matplotlib.pyplot as plt
import numpy as np

tok_stats = []
for i in range(100):
    ex = ds['test'][i]
    w = ex['tokens']
    enc = tokenizer(w, is_split_into_words=True)
    n_w = len(w)
    n_t = sum(1 for wid in enc.word_ids() if wid is not None)
    ratio = n_t / n_w if n_w > 0 else 1.0
    has_ent = any(t != 0 for t in ex['ner_tags'])
    tok_stats.append({'ratio': ratio, 'has_entity': has_ent, 'words': n_w, 'tokens': n_t})

r_all = [s['ratio'] for s in tok_stats]
r_ent = [s['ratio'] for s in tok_stats if s['has_entity']]
r_no_ent = [s['ratio'] for s in tok_stats if not s['has_entity']]

print(f"Overall average R (tokens/words): {np.mean(r_all):.4f}")
print(f"Sentences with entities R:        {np.mean(r_ent):.4f}")
print(f"Sentences without entities R:     {np.mean(r_no_ent):.4f}")

plt.figure(figsize=(7, 4))
plt.bar(['With Entities', 'Without Entities', 'Overall'], [np.mean(r_ent), np.mean(r_no_ent), np.mean(r_all)],
        color=['#3b75af', '#5893d4', '#2b5c8f'], edgecolor='black', width=0.5)
plt.ylabel('Subword / Word Ratio (R)')
plt.title('Subword Expansion Ratio: Entity vs Non-Entity')
plt.ylim(1.0, 1.7)
for i, v in enumerate([np.mean(r_ent), np.mean(r_no_ent), np.mean(r_all)]):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()""")
    ]

    nb.cells = cells
    with open("tokenizer_alignment.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created tokenizer_alignment.ipynb")


def create_finetuning_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    }

    cells = [
        make_clean_cell("markdown", "# Lab 09 — Named Entity Recognition & Information Extraction\n\nHuấn luyện và đánh giá mô hình BERT Token Classification trên tập dữ liệu CoNLL-2003, so sánh với Baseline và ứng dụng vào Information Extraction."),
        make_clean_cell("code", """import pandas as pd
from datasets import load_dataset

ds = load_dataset('lhoestq/conll2003')
tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
print("CoNLL-2003 Dataset Splits:", {k: len(ds[k]) for k in ds})"""),
        make_clean_cell("markdown", "## 1. Baseline vs Pretrained Transformer Results"),
        make_clean_cell("code", """results_df = pd.read_csv('results.csv')
results_df"""),
        make_clean_cell("markdown", "## 2. Pretrained Transformer (BERT) Learning Curves"),
        make_clean_cell("code", """from IPython.display import Image, display

display(Image(filename='learning_curves.png'))"""),
        make_clean_cell("markdown", "## 3. Entity-Level Evaluation by Type (PER, ORG, LOC, MISC)"),
        make_clean_cell("code", """entity_df = pd.read_csv('entity_metrics.csv')
entity_df"""),
        make_clean_cell("code", """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(entity_df))
width = 0.25

ax.bar([i - width for i in x], entity_df['Precision'], width=width, label='Precision', color='#2b5c8f')
ax.bar([i for i in x], entity_df['Recall'], width=width, label='Recall', color='#3b75af')
ax.bar([i + width for i in x], entity_df['F1'], width=width, label='F1', color='#5893d4')

ax.set_xticks(x)
ax.set_xticklabels(entity_df['Entity'])
ax.set_ylabel('Score')
ax.set_ylim(0.6, 1.0)
ax.set_title('Entity-Level Performance by Entity Type (Test Set)')
ax.legend()
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()"""),
        make_clean_cell("markdown", "## 4. Entity Length Analysis (Experiment 5)"),
        make_clean_cell("code", """display(Image(filename='entity_length_analysis.png'))"""),
        make_clean_cell("markdown", "## 5. Token-Level Confusion Matrix"),
        make_clean_cell("code", """display(Image(filename='confusion_matrix.png'))"""),
        make_clean_cell("markdown", "## 6. Qualitative Error Analysis (Experiment 6)"),
        make_clean_cell("code", """import json

with open('error_samples.json') as f:
    errors = json.load(f)

for category in ['false_positives', 'false_negatives', 'boundary_errors']:
    print(f"\\n=== {category.upper().replace('_', ' ')} (Top 3) ===")
    for item in errors[category][:3]:
        print(f"Context: {item['text']}")
        print(f"  Gold: {item['gold']}  |  Pred: {item['pred']}")"""),
        make_clean_cell("markdown", "## 7. Context Analysis: Ambiguous Surface Forms"),
        make_clean_cell("code", """with open('context_analysis.json') as f:
    ambiguous = json.load(f)

for surface, occurrences in list(ambiguous.items())[:3]:
    print(f"\\nSurface Form: '{surface}'")
    for occ in occurrences[:2]:
        print(f"  [{occ['type']}] {occ['context']}")"""),
        make_clean_cell("markdown", "## 8. Application: Information Extraction Pipeline"),
        make_clean_cell("code", """from information_extraction import InformationExtractor

ie = InformationExtractor()

sample_texts = [
    "Barack Obama visited Paris yesterday.",
    "Microsoft and Google signed a partnership in New York.",
    "The United Nations held an emergency summit in Geneva on Friday."
]

for text in sample_texts:
    res = ie.extract_entities(text)
    print(f"\\nText:   '{text}'")
    print(f"Output: {res}")""")
    ]

    nb.cells = cells
    with open("ner_finetuning.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created ner_finetuning.ipynb")


if __name__ == '__main__':
    create_tokenizer_notebook()
    create_finetuning_notebook()
