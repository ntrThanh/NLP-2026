#!/usr/bin/env python3
"""Lab 10 — analysis of the fine-tuned QA model vs the baseline.

Produces:
  results.csv                    Baseline vs Transformer + Experiment 1 (context length)
  learning_curves.png            train/val loss, val EM/F1 (Experiment 2)
  em_f1_analysis.png             EM vs F1 per model, per-example F1 distribution (Experiment 3)
  answer_length_analysis.png     EM/F1 by answer length bucket (Experiment 5)
  outputs/analysis.json          all tables (lexical overlap, robustness, boundary stats ...)
  outputs/error_samples.json     examples for error_analysis.md
"""
import collections
import json
import os
import random
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_qa import STOPWORDS, baseline_answer, stem, tokens_with_offsets
from qa_model import evaluate_predictions, load_splits, normalize_answer, squad_em, squad_f1

OUT = 'outputs'
SEED = 42
plt.rcParams.update({'figure.dpi': 130, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
                     'axes.spines.right': False})
C_BASE, C_BERT = '#9aa5b1', '#2f6db5'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def content_stems(text):
    return {stem(t) for t, _, _ in tokens_with_offsets(text) if t.isalnum() and t.lower() not in STOPWORDS}


def length_bucket(n):
    return '1' if n == 1 else '2–3' if n <= 3 else '4–6' if n <= 6 else '7+'


SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')


def split_sentences(text):
    out, pos = [], 0
    for part in SENT_SPLIT.split(text):
        start = text.index(part, pos)
        out.append((start, start + len(part)))
        pos = start + len(part)
    return out


# ════════════════════════════════════════════════════════════════════════
# Per-example table
# ════════════════════════════════════════════════════════════════════════
def per_example(test, bert, base):
    rows = []
    for ex in test:
        golds = ex['answers']['text']
        g_spans = [(s, s + len(t)) for s, t in zip(ex['answers']['answer_start'], golds)]
        p = bert[ex['id']]
        b = base[ex['id']]
        f1s = [squad_f1(p['text'], g) for g in golds]
        k = int(np.argmax(f1s))                     # gold used for boundary comparison
        gs, ge = g_spans[k]
        # sentence containing the (first) gold answer -> lexical overlap with the question
        sents = split_sentences(ex['context'])
        a0 = g_spans[0][0]
        s_start, s_end = next(((s, e) for s, e in sents if s <= a0 < e), sents[0])
        ans_sent = ex['context'][s_start:s_end]
        sent_wo_ans = ans_sent.replace(golds[0], ' ')
        q_c = content_stems(ex['question'])
        overlap = len(q_c & content_stems(sent_wo_ans)) / max(1, len(q_c))
        rows.append({
            'id': ex['id'], 'title': ex['title'], 'question': ex['question'], 'golds': golds,
            'gold': golds[k], 'gold_start': gs, 'gold_end': ge,
            'pred': p['text'], 'pred_start': p['start_char'], 'pred_end': p['end_char'],
            'em': max(squad_em(p['text'], g) for g in golds), 'f1': max(f1s),
            'base_pred': b['text'], 'base_em': max(squad_em(b['text'], g) for g in golds),
            'base_f1': max(squad_f1(b['text'], g) for g in golds),
            'ans_len': len(golds[0].split()), 'len_bucket': length_bucket(len(golds[0].split())),
            'lex_overlap': overlap, 'context': ex['context'], 'answer_sentence': ans_sent,
        })
    return pd.DataFrame(rows)


def boundary_category(r):
    """Classify a prediction against the best-matching gold character span."""
    if r['em'] == 1:
        return 'correct'
    ps, pe, gs, ge = r['pred_start'], r['pred_end'], r['gold_start'], r['gold_end']
    # tolerate whitespace / trailing punctuation differences at the edges
    same_s = normalize_answer(r['context'][min(ps, gs):max(ps, gs)]) == ''
    same_e = normalize_answer(r['context'][min(pe, ge):max(pe, ge)]) == ''
    if pe <= gs or ps >= ge:
        return 'wrong span (no overlap)'
    if same_s and not same_e:
        return 'wrong end — too long' if pe > ge else 'wrong end — too short'
    if same_e and not same_s:
        return 'wrong start — too long' if ps < gs else 'wrong start — too short'
    if ps >= gs and pe <= ge:
        return 'both wrong — inside gold (partial)'
    if ps <= gs and pe >= ge:
        return 'both wrong — covers gold (over-extended)'
    return 'both wrong — shifted overlap'


# ════════════════════════════════════════════════════════════════════════
# Robustness (Prediction 3 & 4): distractor / irrelevant paragraph / shuffled sentences
# ════════════════════════════════════════════════════════════════════════
WH_RE = re.compile(r'\b(what|which|who|whom|whose|when|where|how many|how much|how)\b', re.I)


def make_distractor(question, fake_answer):
    """AddSent-lite: turn the question into a statement containing a fake answer."""
    q = question.strip().rstrip('?').strip()
    if not WH_RE.search(q):
        return f'{q} {fake_answer}.'
    s = WH_RE.sub(fake_answer, q, count=1)
    return s[0].upper() + s[1:] + '.'


CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")
NUM_RE = re.compile(r"\b\d[\d,.]*\b")


def perturb_question(question, donor_context, rng):
    """AddSent-style: replace proper nouns (non-initial capitalised words) and numbers in the question
    with ones taken from an unrelated article, so the distractor no longer answers the question."""
    donors = [w for w in CAP_RE.findall(donor_context) if w.lower() not in STOPWORDS] or ['Atlantis']
    changed = [False]

    def cap(m):
        if m.start() == 0 or m.group().lower() in STOPWORDS or m.group().isupper():
            return m.group()
        changed[0] = True
        return rng.choice(donors)

    def num(m):
        changed[0] = True
        return str(int(re.sub(r'\D', '', m.group()) or 0) + rng.randint(3, 40))
    q = CAP_RE.sub(cap, question)
    q = NUM_RE.sub(num, q)
    return q, changed[0]


def perturb(test_subset, all_test, mode, rng):
    titles = list(all_test['title'])
    contexts = list(all_test['context'])
    answers = [a['text'][0] for a in all_test['answers']]
    out = []
    for ex in test_subset:
        ctx = ex['context']
        distractor = ''
        if mode == 'original':
            new = ctx
        elif mode in ('irrelevant_append', 'irrelevant_prepend'):
            while True:
                j = rng.randrange(len(contexts))
                if titles[j] != ex['title']:
                    break
            new = ctx + ' ' + contexts[j] if mode == 'irrelevant_append' else contexts[j] + ' ' + ctx
        elif mode in ('distractor_contradictory', 'distractor_addsent'):
            while True:
                j = rng.randrange(len(answers))
                if titles[j] != ex['title'] and normalize_answer(answers[j]) not in \
                        {normalize_answer(g) for g in ex['answers']['text']}:
                    break
            q = ex['question']
            if mode == 'distractor_addsent':
                q, _ = perturb_question(q, contexts[j], rng)
            distractor = make_distractor(q, answers[j])
            new = ctx + ' ' + distractor
        elif mode == 'shuffled_sentences':
            sents = [ctx[s:e] for s, e in split_sentences(ctx)]
            if len(sents) > 1:
                orig = list(sents)
                for _ in range(10):
                    rng.shuffle(sents)
                    if sents != orig:
                        break
            new = ' '.join(sents)
        out.append({'id': ex['id'], 'title': ex['title'], 'question': ex['question'], 'context': new,
                    'answers': ex['answers'], 'distractor': distractor})
    return out


def robustness(test, model_dir, n=1000):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from qa_model import postprocess_predictions, predict_logits, prepare_eval_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device).eval()
    rng = random.Random(SEED)
    idx = rng.sample(range(len(test)), n)
    subset = test.select(idx)
    rows, examples_out = [], {}
    for mode in ('original', 'irrelevant_append', 'irrelevant_prepend', 'distractor_addsent',
                 'distractor_contradictory', 'shuffled_sentences'):
        data = perturb(subset, test, mode, random.Random(SEED))
        ds = Dataset.from_list(data)
        feats = ds.map(prepare_eval_features, batched=True, remove_columns=ds.column_names,
                       fn_kwargs=dict(tokenizer=tok, max_length=384, stride=128))
        s, e = predict_logits(model, feats, device, batch_size=32)
        preds = postprocess_predictions(ds, feats, s, e)
        m = evaluate_predictions({k: v['text'] for k, v in preds.items()}, data)
        bpreds = {d['id']: baseline_answer(d['question'], d['context'])['text'] for d in data}
        bm = evaluate_predictions(bpreds, data)
        rows.append({'condition': mode, 'n': n, 'windows/example': round(len(feats) / n, 3),
                     'bert_em': round(m['exact_match'], 2), 'bert_f1': round(m['f1'], 2),
                     'baseline_em': round(bm['exact_match'], 2), 'baseline_f1': round(bm['f1'], 2)})
        print(rows[-1], flush=True)
        if mode.startswith('distractor'):
            # examples where the distractor flipped a correct answer
            orig = examples_out['original_preds']
            flips = []
            for d in data:
                g = d['answers']['text']
                if max(squad_em(orig[d['id']], x) for x in g) == 1 and \
                        max(squad_em(preds[d['id']]['text'], x) for x in g) == 0:
                    flips.append({'question': d['question'], 'gold': g[0], 'pred_original': orig[d['id']],
                                  'pred_with_distractor': preds[d['id']]['text'],
                                  'distractor': d['distractor']})
            examples_out[f'{mode}_flips'] = flips[:10]
            examples_out[f'{mode}_flip_rate'] = 100.0 * len(flips) / n
        if mode == 'original':
            examples_out['original_preds'] = {k: v['text'] for k, v in preds.items()}
    examples_out.pop('original_preds')
    return rows, examples_out


# ════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════
def plot_learning_curves(hist, path):
    h = pd.DataFrame(hist)
    h = h[~(h['epoch'].diff(-1).abs() < 0.01)]      # drop an eval that sits right before the final one
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(h['epoch'], h['train_loss'], 'o-', color=C_BERT, label='train loss (running mean)')
    ax[0].plot(h['epoch'], h['val_loss'], 's--', color='#d9822b', label='validation loss')
    ax[0].set(title='Loss  L = (L_start + L_end) / 2', xlabel='epoch', ylabel='loss')
    ax[0].legend()
    ax[1].plot(h['epoch'], h['train_start_loss'], 'o-', color='#3b8f5a', label='train start loss')
    ax[1].plot(h['epoch'], h['train_end_loss'], 'o-', color='#b5452f', label='train end loss')
    ax[1].set(title='Start vs end loss (train)', xlabel='epoch')
    ax[1].legend()
    ax[2].plot(h['epoch'], h['val_em'], 'o-', color='#7b4bb5', label='val EM')
    ax[2].plot(h['epoch'], h['val_f1'], 's-', color=C_BERT, label='val F1')
    for x, y in zip(h['epoch'], h['val_f1']):
        ax[2].annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)
    ax[2].set(title='Validation EM / F1 (held-out dev articles)', xlabel='epoch', ylabel='%')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_em_f1(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    models = ['Baseline', 'BERT fine-tuned']
    ems = [100 * df['base_em'].mean(), 100 * df['em'].mean()]
    f1s = [100 * df['base_f1'].mean(), 100 * df['f1'].mean()]
    x = np.arange(2)
    b1 = ax[0].bar(x - 0.18, ems, 0.36, label='EM', color='#7b4bb5')
    b2 = ax[0].bar(x + 0.18, f1s, 0.36, label='F1', color=C_BERT)
    ax[0].bar_label(b1, fmt='%.1f', fontsize=8)
    ax[0].bar_label(b2, fmt='%.1f', fontsize=8)
    ax[0].set_xticks(x, models)
    ax[0].set(title='EM vs F1 — test = SQuAD v1.1 dev (10,570 q)', ylabel='%', ylim=(0, 100))
    ax[0].legend()

    def outcome(em, f1):
        return np.select([em == 1, f1 > 0], ['EM = 1', 'EM = 0, F1 > 0'], 'F1 = 0')
    cats = ['EM = 1', 'EM = 0, F1 > 0', 'F1 = 0']
    colors = ['#3b8f5a', '#e0a23b', '#b5452f']
    for i, (em_col, f1_col) in enumerate([('base_em', 'base_f1'), ('em', 'f1')]):
        o = pd.Series(outcome(df[em_col].values, df[f1_col].values)).value_counts(normalize=True) * 100
        left = 0
        for c, col in zip(cats, colors):
            v = o.get(c, 0)
            ax[1].barh(i, v, left=left, color=col, label=c if i == 0 else None)
            if v > 4:
                ax[1].text(left + v / 2, i, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=8)
            left += v
    ax[1].set_yticks([0, 1], models)
    ax[1].set(title='Outcome per question', xlabel='% of questions', xlim=(0, 100))
    ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=8)

    partial = df[(df['em'] == 0) & (df['f1'] > 0)]['f1']
    ax[2].hist(partial, bins=20, range=(0, 1), color='#e0a23b', edgecolor='white')
    ax[2].set(title=f'BERT: F1 distribution when EM = 0, F1 > 0\n(n = {len(partial)})',
              xlabel='token F1', ylabel='questions')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_answer_length(tab, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(tab))
    for a, metric in zip(ax, ['EM', 'F1']):
        b1 = a.bar(x - 0.18, tab[f'baseline_{metric}'], 0.36, color=C_BASE, label='Baseline')
        b2 = a.bar(x + 0.18, tab[f'bert_{metric}'], 0.36, color=C_BERT, label='BERT fine-tuned')
        a.bar_label(b1, fmt='%.1f', fontsize=8)
        a.bar_label(b2, fmt='%.1f', fontsize=8)
        a.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(tab['answer_length'], tab['n'])])
        a.set(title=f'{metric} by gold answer length (words)', ylabel='%', ylim=(0, 100))
        a.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
def main(run_robustness=True):
    _, _, test = load_splits()
    bert = load_json(f'{OUT}/main/test_predictions.json')
    base = load_json(f'{OUT}/baseline/test_predictions.json')
    df = per_example(test, bert, base)
    df['category'] = df.apply(boundary_category, axis=1)
    analysis = {}

    # ── results.csv ────────────────────────────────────────────────────
    main_s = load_json(f'{OUT}/main/summary.json')
    rows = [
        {'experiment': 'Exp3: model comparison', 'model': 'Baseline (keyword overlap + answer-type rules)',
         'max_length': '-', 'stride': '-', 'train_examples': 0, 'epochs': 0,
         'EM': round(100 * df['base_em'].mean(), 2), 'F1': round(100 * df['base_f1'].mean(), 2),
         'train_time_min': 0},
        {'experiment': 'Exp2/3: model comparison', 'model': 'bert-base-cased fine-tuned',
         'max_length': 384, 'stride': 128, 'train_examples': main_s['train_examples'],
         'epochs': main_s['config']['epochs'], 'EM': round(main_s['test']['exact_match'], 2),
         'F1': round(main_s['test']['f1'], 2), 'train_time_min': round(main_s['train_time_sec'] / 60, 1)},
    ]
    for L in (128, 256, 384):
        p = f'{OUT}/len{L}/summary.json'
        if os.path.exists(p):
            s = load_json(p)
            rows.append({'experiment': 'Exp1: context length', 'model': 'bert-base-cased fine-tuned',
                         'max_length': L, 'stride': s['config']['stride'], 'train_examples': s['train_examples'],
                         'epochs': s['config']['epochs'], 'EM': round(s['test']['exact_match'], 2),
                         'F1': round(s['test']['f1'], 2), 'train_time_min': round(s['train_time_sec'] / 60, 1)})
    pd.DataFrame(rows).to_csv('results.csv', index=False)
    print(pd.DataFrame(rows).to_string())

    # ── learning curves / EM-F1 ────────────────────────────────────────
    plot_learning_curves(load_json(f'{OUT}/main/history.json'), 'learning_curves.png')
    plot_em_f1(df, 'em_f1_analysis.png')

    # ── answer length (Exp 5) ──────────────────────────────────────────
    order = ['1', '2–3', '4–6', '7+']
    tab = df.groupby('len_bucket').agg(n=('id', 'size'), bert_EM=('em', 'mean'), bert_F1=('f1', 'mean'),
                                       baseline_EM=('base_em', 'mean'), baseline_F1=('base_f1', 'mean'))
    tab = tab.reindex(order)
    for c in ['bert_EM', 'bert_F1', 'baseline_EM', 'baseline_F1']:
        tab[c] = (100 * tab[c]).round(2)
    tab['bert_F1_minus_EM'] = (tab['bert_F1'] - tab['bert_EM']).round(2)
    tab = tab.reset_index().rename(columns={'len_bucket': 'answer_length'})
    plot_answer_length(tab, 'answer_length_analysis.png')
    analysis['answer_length'] = tab.to_dict(orient='records')

    # ── lexical overlap (Exp 4) ────────────────────────────────────────
    q1, q3 = df['lex_overlap'].quantile([0.25, 0.75])
    df['overlap_group'] = np.select([df['lex_overlap'] <= q1, df['lex_overlap'] >= q3],
                                    [f'low overlap (≤{q1:.2f})', f'high overlap (≥{q3:.2f})'], 'middle')
    lex = df.groupby('overlap_group').agg(n=('id', 'size'), mean_overlap=('lex_overlap', 'mean'),
                                          bert_EM=('em', 'mean'), bert_F1=('f1', 'mean'),
                                          baseline_EM=('base_em', 'mean'), baseline_F1=('base_f1', 'mean'))
    for c in ['bert_EM', 'bert_F1', 'baseline_EM', 'baseline_F1']:
        lex[c] = (100 * lex[c]).round(2)
    lex['mean_overlap'] = lex['mean_overlap'].round(3)
    analysis['lexical_overlap'] = lex.reset_index().to_dict(orient='records')
    print(lex)

    # ── boundary stats (wrong start / wrong end) ───────────────────────
    wrong = df[df['em'] == 0]
    cat_counts = df['category'].value_counts()
    same_s = wrong.apply(lambda r: normalize_answer(r['context'][min(r['pred_start'], r['gold_start']):
                                                                 max(r['pred_start'], r['gold_start'])]) == '', axis=1)
    same_e = wrong.apply(lambda r: normalize_answer(r['context'][min(r['pred_end'], r['gold_end']):
                                                                 max(r['pred_end'], r['gold_end'])]) == '', axis=1)
    overl = ~((wrong['pred_end'] <= wrong['gold_start']) | (wrong['pred_start'] >= wrong['gold_end']))
    analysis['boundary'] = {
        'n_test': len(df), 'n_em0': int(len(wrong)),
        'n_wrong_start': int((~same_s).sum()), 'n_wrong_end': int((~same_e).sum()),
        'n_wrong_start_among_overlapping': int((~same_s & overl).sum()),
        'n_wrong_end_among_overlapping': int((~same_e & overl).sum()),
        'n_only_start_wrong': int((~same_s & same_e).sum()), 'n_only_end_wrong': int((same_s & ~same_e).sum()),
        'n_both_wrong_overlapping': int((~same_s & ~same_e & overl).sum()),
        'n_no_overlap': int((~overl).sum()),
        'categories': {k: int(v) for k, v in cat_counts.items()},
        'em0_f1_gt0': int(((df['em'] == 0) & (df['f1'] > 0)).sum()),
        'mean_pred_len_words_when_wrong': float(wrong['pred'].str.split().str.len().mean()),
        'mean_gold_len_words_when_wrong': float(wrong['gold'].str.split().str.len().mean()),
    }
    print(json.dumps(analysis['boundary'], indent=1))

    # ── baseline vs bert agreement ─────────────────────────────────────
    analysis['agreement'] = {
        'both_em': int(((df['em'] == 1) & (df['base_em'] == 1)).sum()),
        'bert_only_em': int(((df['em'] == 1) & (df['base_em'] == 0)).sum()),
        'baseline_only_em': int(((df['em'] == 0) & (df['base_em'] == 1)).sum()),
        'neither_em': int(((df['em'] == 0) & (df['base_em'] == 0)).sum()),
    }

    # ── error samples for the write-up ─────────────────────────────────
    rng = np.random.RandomState(SEED)

    def sample(frame, k):
        frame = frame.sample(min(k, len(frame)), random_state=rng)
        cols = ['id', 'question', 'golds', 'pred', 'em', 'f1', 'category', 'len_bucket', 'lex_overlap',
                'answer_sentence', 'pred_start', 'gold_start']
        recs = frame[cols].to_dict(orient='records')
        for r, (_, full) in zip(recs, frame.iterrows()):
            ctx = full['context']
            lo, hi = max(0, min(full['pred_start'], full['gold_start']) - 150), \
                max(full['pred_end'], full['gold_end']) + 150
            r['context_window'] = ('…' if lo > 0 else '') + ctx[lo:hi] + ('…' if hi < len(ctx) else '')
        return recs

    samples = {}
    for cat in df['category'].unique():
        if cat != 'correct':
            samples[cat] = sample(df[df['category'] == cat], 10)
    samples['em0_f1_gt0'] = sample(df[(df['em'] == 0) & (df['f1'] > 0)], 12)
    samples['low_overlap_wrong'] = sample(df[(df['overlap_group'].str.startswith('low')) & (df['em'] == 0)], 8)
    samples['baseline_only_correct'] = sample(df[(df['em'] == 0) & (df['base_em'] == 1)], 6)

    # ── robustness (Prediction 3/4) ────────────────────────────────────
    if run_robustness and os.path.exists(f'{OUT}/main/model'):
        rob, rob_ex = robustness(test, f'{OUT}/main/model')
        analysis['robustness'] = rob
        for k, v in rob_ex.items():
            (analysis if k.endswith('rate') else samples)[k] = v

    with open(f'{OUT}/analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    with open(f'{OUT}/error_samples.json', 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False, default=float)
    df.drop(columns=['context']).to_csv(f'{OUT}/per_example.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
