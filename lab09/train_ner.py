#!/usr/bin/env python3
import json
import os
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score as seq_p
from seqeval.metrics import recall_score as seq_r
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SEED = 42
MODEL_NAME = 'bert-base-cased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ID2TAG = {i: t for i, t in enumerate(TAG_NAMES)}
TAG2ID = {t: i for i, t in enumerate(TAG_NAMES)}
NUM_LABELS = len(TAG_NAMES)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Tags: {TAG_NAMES}")

# ── Load Dataset ────────────────────────────────────────────────────────
print("\nLoading CoNLL-2003 dataset...")
ds = load_dataset('lhoestq/conll2003')
print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── 1. Baseline Model (Logistic Regression) ────────────────────────────
print("\n--- Running Baseline (Logistic Regression) ---")


def extract_features(word, idx, sentence):
    feats = {
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'has_hyphen': '-' in word,
        'has_dot': '.' in word,
        'prefix1': word[0],
        'prefix2': word[:2],
        'suffix1': word[-1],
        'BOS': idx == 0,
        'EOS': idx == len(sentence) - 1,
    }
    if idx > 0:
        feats['prev_word.lower'] = sentence[idx - 1].lower()
        feats['prev_word.istitle'] = sentence[idx - 1].istitle()
    else:
        feats['prev_word.lower'] = '<BOS>'
        feats['prev_word.istitle'] = False

    if idx < len(sentence) - 1:
        feats['next_word.lower'] = sentence[idx + 1].lower()
        feats['next_word.istitle'] = sentence[idx + 1].istitle()
    else:
        feats['next_word.lower'] = '<EOS>'
        feats['next_word.istitle'] = False

    return feats


def sentence_to_features(sentence):
    return [extract_features(w, i, sentence) for i, w in enumerate(sentence)]


train_feats = []
train_labels = []
for ex in ds['train']:
    sf = sentence_to_features(ex['tokens'])
    train_feats.extend(sf)
    train_labels.extend([ID2TAG[t] for t in ex['ner_tags']])

vocab = {}
for feats in train_feats:
    for k, v in feats.items():
        tok = f"{k}={v}"
        if tok not in vocab:
            vocab[tok] = len(vocab)


def vectorize(feature_list, vocab_dict):
    from scipy.sparse import dok_matrix
    X = dok_matrix((len(feature_list), len(vocab_dict)), dtype=np.float32)
    for i, feats in enumerate(feature_list):
        for k, v in feats.items():
            tok = f"{k}={v}"
            if tok in vocab_dict:
                X[i, vocab_dict[tok]] = 1.0
    return X.tocsr()


X_train = vectorize(train_feats, vocab)
le = LabelEncoder()
le.fit(TAG_NAMES)
y_train = le.transform(train_labels)

baseline_clf = LogisticRegression(max_iter=150, solver='saga', C=1.0, n_jobs=-1, random_state=SEED)
baseline_clf.fit(X_train, y_train)

test_gold_seqs = []
baseline_pred_seqs = []
for ex in ds['test']:
    sf = sentence_to_features(ex['tokens'])
    X_test_s = vectorize(sf, vocab)
    preds = le.inverse_transform(baseline_clf.predict(X_test_s))
    gold = [ID2TAG[t] for t in ex['ner_tags']]
    test_gold_seqs.append(gold)
    baseline_pred_seqs.append(list(preds))

base_p = seq_p(test_gold_seqs, baseline_pred_seqs)
base_r = seq_r(test_gold_seqs, baseline_pred_seqs)
base_f1 = seq_f1(test_gold_seqs, baseline_pred_seqs)
print(f"Baseline - Precision: {base_p:.4f}, Recall: {base_r:.4f}, F1: {base_f1:.4f}")

# ── 2. Pretrained Transformer Fine-tuning ──────────────────────────────
print(f"\n--- Fine-tuning {MODEL_NAME} ---")


class CoNLLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def align_labels_strategy_a(words, tags):
    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_tensors='pt'
    )
    word_ids = enc.word_ids(batch_index=0)
    label_ids = []
    prev_wid = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(-100)
        elif wid != prev_wid:
            label_ids.append(tags[wid])
        else:
            label_ids.append(-100)
        prev_wid = wid

    return {
        'input_ids': enc['input_ids'].squeeze(0),
        'attention_mask': enc['attention_mask'].squeeze(0),
        'labels': torch.tensor(label_ids, dtype=torch.long)
    }


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        res = align_labels_strategy_a(item['tokens'], item['ner_tags'])
        input_ids.append(res['input_ids'])
        attention_mask.append(res['attention_mask'])
        labels.append(res['labels'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }


train_loader = DataLoader(CoNLLDataset(ds['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(CoNLLDataset(ds['validation']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(CoNLLDataset(ds['test']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


def eval_model(loader):
    model.eval()
    total_loss = 0.0
    true_seqs = []
    pred_seqs = []
    with torch.no_grad():
        for batch in loader:
            b_ids = batch['input_ids'].to(DEVICE)
            b_mask = batch['attention_mask'].to(DEVICE)
            b_labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            total_loss += outputs.loss.item()

            logits = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels_np = b_labels.cpu().numpy()

            for p_row, l_row in zip(logits, labels_np):
                p_tokens = []
                l_tokens = []
                for p, l in zip(p_row, l_row):
                    if l != -100:
                        p_tokens.append(ID2TAG[p])
                        l_tokens.append(ID2TAG[l])
                true_seqs.append(l_tokens)
                pred_seqs.append(p_tokens)

    avg_loss = total_loss / len(loader)
    f1 = seq_f1(true_seqs, pred_seqs)
    return avg_loss, f1, true_seqs, pred_seqs


train_losses = []
val_losses = []
val_f1s = []

print("Training progress:")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_loader):
        b_ids = batch['input_ids'].to(DEVICE)
        b_mask = batch['attention_mask'].to(DEVICE)
        b_labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss, val_f1, _, _ = eval_model(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1s.append(val_f1)
    print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")

# ── 3. Test Evaluation & Metrics ───────────────────────────────────────
print("\n--- Evaluating on Test Set ---")
test_loss, trans_f1, test_true_seqs, test_pred_seqs = eval_model(test_loader)
trans_p = seq_p(test_true_seqs, test_pred_seqs)
trans_r = seq_r(test_true_seqs, test_pred_seqs)

print(f"Transformer - Precision: {trans_p:.4f}, Recall: {trans_r:.4f}, F1: {trans_f1:.4f}")
print("\nClassification Report (Entity Level):")
print(seq_report(test_true_seqs, test_pred_seqs, digits=4))

# Save results.csv
results_df = pd.DataFrame([
    {
        'Model': 'Baseline (Logistic Regression)',
        'Precision': round(base_p, 4),
        'Recall': round(base_r, 4),
        'F1': round(base_f1, 4),
        'Architecture': 'Handcrafted Orthographic Features + SAGA'
    },
    {
        'Model': f'Pretrained Transformer ({MODEL_NAME})',
        'Precision': round(trans_p, 4),
        'Recall': round(trans_r, 4),
        'F1': round(trans_f1, 4),
        'Architecture': 'BERT-Encoder + Linear Token Classifier'
    }
])
results_df.to_csv('results.csv', index=False)
print("Saved results.csv")

# Save entity_metrics.csv
report_dict = seq_report(test_true_seqs, test_pred_seqs, digits=4, output_dict=True)
entity_rows = []
for etype in ['PER', 'ORG', 'LOC', 'MISC']:
    if etype in report_dict:
        d = report_dict[etype]
        entity_rows.append({
            'Entity': etype,
            'Precision': round(d['precision'], 4),
            'Recall': round(d['recall'], 4),
            'F1': round(d['f1-score'], 4),
            'Support': int(d['support'])
        })
entity_df = pd.DataFrame(entity_rows)
entity_df.to_csv('entity_metrics.csv', index=False)
print("Saved entity_metrics.csv")

# ── 4. Visualizations ─────────────────────────────────────────────────
# 4.1 Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
epochs_range = list(range(1, EPOCHS + 1))
ax1.plot(epochs_range, train_losses, 'o-', color='#1f77b4', label='Train Loss')
ax1.plot(epochs_range, val_losses, 's--', color='#ff7f0e', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Cross-Entropy Loss')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(epochs_range, val_f1s, 'D-', color='#2ca02c', label='Validation F1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.set_title('Validation Entity-Level F1')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=200)
plt.close()

# 4.2 Confusion Matrix (Token Level)
flat_true = [t for seq in test_true_seqs for t in seq]
flat_pred = [p for seq in test_pred_seqs for p in seq]
labels_for_cm = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
cm = confusion_matrix(flat_true, flat_pred, labels=labels_for_cm)

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_for_cm)
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.title('Token-Level Confusion Matrix (Test Set)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=200)
plt.close()
print("Saved confusion_matrix.png")


# ── 5. Experiment 5: Entity Length Analysis ───────────────────────────
def extract_entities(tag_seq):
    entities = []
    cur_type = None
    start = None
    for i, tag in enumerate(tag_seq):
        if tag.startswith('B-'):
            if cur_type is not None:
                entities.append((cur_type, start, i))
            cur_type = tag[2:]
            start = i
        elif tag.startswith('I-'):
            if cur_type != tag[2:]:
                if cur_type is not None:
                    entities.append((cur_type, start, i))
                cur_type = tag[2:]
                start = i
        else:
            if cur_type is not None:
                entities.append((cur_type, start, i))
                cur_type = None
                start = None
    if cur_type is not None:
        entities.append((cur_type, start, len(tag_seq)))
    return entities


len_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
for g_seq, p_seq in zip(test_true_seqs, test_pred_seqs):
    g_ents = set(extract_entities(g_seq))
    p_ents = set(extract_entities(p_seq))

    for ent in g_ents:
        l = ent[2] - ent[1]
        b = str(l) if l <= 3 else '4+'
        if ent in p_ents:
            len_stats[b]['tp'] += 1
        else:
            len_stats[b]['fn'] += 1

    for ent in p_ents:
        l = ent[2] - ent[1]
        b = str(l) if l <= 3 else '4+'
        if ent not in g_ents:
            len_stats[b]['fp'] += 1

len_order = ['1', '2', '3', '4+']
f1_by_len = {}
for b in len_order:
    tp = len_stats[b]['tp']
    fp = len_stats[b]['fp']
    fn = len_stats[b]['fn']
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    f1_by_len[b] = round(f, 4)
    print(f"Length {b}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f} (TP={tp}, FP={fp}, FN={fn})")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(len_order, [f1_by_len[k] for k in len_order], color=['#2b5c8f', '#3b75af', '#5893d4', '#84b6f4'], edgecolor='black', linewidth=0.8)
ax.set_xlabel('Entity Length (Number of Tokens)')
ax.set_ylabel('Entity-Level F1 Score')
ax.set_ylim(0, 1.05)
ax.set_title('Entity-Level F1 by Entity Length')
ax.grid(axis='y', linestyle=':', alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('entity_length_analysis.png', dpi=200)
plt.close()
print("Saved entity_length_analysis.png")

# ── 6. Tokenization Analysis (Experiment 1) ───────────────────────────
tok_analysis = []
entity_ratios = []
non_entity_ratios = []
split_examples = []

for i, ex in enumerate(ds['test']):
    if i >= 50:
        break
    words = ex['tokens']
    tags = ex['ner_tags']
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    word_ids = [w for w in enc.word_ids(batch_index=0) if w is not None]

    num_words = len(words)
    num_tokens = len(word_ids)
    ratio = num_tokens / num_words if num_words > 0 else 1.0
    has_ent = any(t != 0 for t in tags)

    if has_ent:
        entity_ratios.append(ratio)
    else:
        non_entity_ratios.append(ratio)

    # find words split into subwords
    counts = Counter(word_ids)
    for wid, c in counts.items():
        if c > 1 and len(split_examples) < 15:
            split_examples.append({
                'word': words[wid],
                'subwords': [tokenizer.convert_ids_to_tokens(enc['input_ids'][j]) for j, w in enumerate(enc.word_ids(batch_index=0)) if w == wid],
                'tag': ID2TAG[tags[wid]]
            })

tok_summary = {
    'avg_overall_ratio': round(float(np.mean(entity_ratios + non_entity_ratios)), 4),
    'avg_entity_ratio': round(float(np.mean(entity_ratios)), 4) if entity_ratios else 1.0,
    'avg_non_entity_ratio': round(float(np.mean(non_entity_ratios)), 4) if non_entity_ratios else 1.0,
    'sample_splits': split_examples
}
with open('tokenization_stats.json', 'w', encoding='utf-8') as f:
    json.dump(tok_summary, f, indent=2, ensure_ascii=False)
print("Saved tokenization_stats.json")

# ── 7. Error Analysis Data (Experiment 6) ─────────────────────────────
error_records = {'false_positives': [], 'false_negatives': [], 'boundary_errors': []}

for sent_idx, (g_seq, p_seq) in enumerate(zip(test_true_seqs, test_pred_seqs)):
    words = ds['test'][sent_idx]['tokens']
    g_ents = extract_entities(g_seq)
    p_ents = extract_entities(p_seq)

    g_set = set(g_ents)
    p_set = set(p_ents)

    # False positives & Boundary errors
    for p_ent in p_ents:
        if p_ent not in g_set:
            p_type, p_s, p_e = p_ent
            matched_boundary = False
            for g_ent in g_ents:
                g_type, g_s, g_e = g_ent
                if (max(p_s, g_s) < min(p_e, g_e)):  # overlaps
                    matched_boundary = True
                    if len(error_records['boundary_errors']) < 8:
                        error_records['boundary_errors'].append({
                            'text': " ".join(words[max(0, min(p_s, g_s) - 2): min(len(words), max(p_e, g_e) + 2)]),
                            'sentence': " ".join(words),
                            'gold': f"{g_type}: {' '.join(words[g_s:g_e])}",
                            'pred': f"{p_type}: {' '.join(words[p_s:p_e])}",
                            'error_type': 'wrong_boundary' if g_type == p_type else 'wrong_boundary_and_type'
                        })
                    break
            if not matched_boundary and len(error_records['false_positives']) < 8:
                error_records['false_positives'].append({
                    'text': " ".join(words[max(0, p_s - 2): min(len(words), p_e + 2)]),
                    'sentence': " ".join(words),
                    'gold': 'O',
                    'pred': f"{p_type}: {' '.join(words[p_s:p_e])}",
                    'error_type': 'spurious_entity'
                })

    # False negatives
    for g_ent in g_ents:
        if g_ent not in p_set:
            g_type, g_s, g_e = g_ent
            overlaps = any(max(g_s, pe[1]) < min(g_e, pe[2]) for pe in p_ents)
            if not overlaps and len(error_records['false_negatives']) < 8:
                error_records['false_negatives'].append({
                    'text': " ".join(words[max(0, g_s - 2): min(len(words), g_e + 2)]),
                    'sentence': " ".join(words),
                    'gold': f"{g_type}: {' '.join(words[g_s:g_e])}",
                    'pred': 'O',
                    'error_type': 'missing_entity'
                })

with open('error_samples.json', 'w', encoding='utf-8') as f:
    json.dump(error_records, f, indent=2, ensure_ascii=False)
print("Saved error_samples.json")

# ── 8. Context Analysis (Same surface form, different entity) ─────────
surface_map = defaultdict(list)
for idx, ex in enumerate(ds['test']):
    words = ex['tokens']
    tags = [ID2TAG[t] for t in ex['ner_tags']]
    ents = extract_entities(tags)
    for etype, s, e in ents:
        sf = " ".join(words[s:e])
        surface_map[sf].append({
            'type': etype,
            'context': " ".join(words),
            'idx': idx
        })

ambiguous = {}
for sf, occurrences in surface_map.items():
    types = set(o['type'] for o in occurrences)
    if len(types) > 1 and len(occurrences) >= 2:
        ambiguous[sf] = occurrences

with open('context_analysis.json', 'w', encoding='utf-8') as f:
    json.dump({k: v[:4] for k, v in list(ambiguous.items())[:10]}, f, indent=2, ensure_ascii=False)
print(f"Saved context_analysis.json (found {len(ambiguous)} ambiguous surface forms)")

print("\n=== All experiments completed successfully! ===")
