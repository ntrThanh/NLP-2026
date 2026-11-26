#!/usr/bin/env python3
"""Lab 10 — Extractive Question Answering (SQuAD v1.1).

Core library + training CLI:
  * explicit character-span -> token-span alignment through offset_mapping
  * sliding-window tokenization (truncation="only_second", stride, overflow)
  * start/end loss written out explicitly: L = (CE_start + CE_end) / 2
  * n-best span decoding aggregated across windows
  * SQuAD EM / token-level F1 implemented from scratch

Usage (server22):
  python qa_model.py --tag main --max_length 384 --stride 128 --epochs 2 --save_model
  python qa_model.py --tag len128 --max_length 128 --stride 42 --train_samples 20000 --epochs 1
"""
import argparse
import collections
import json
import math
import os
import random
import re
import string
import time

import numpy as np

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODEL_NAME = 'bert-base-cased'
SEED = 42
DEV_SIZE = 2000          # examples held out from SQuAD-train (split by article title)
N_BEST = 20
MAX_ANSWER_TOKENS = 30


# ════════════════════════════════════════════════════════════════════════
# 1. Metrics (SQuAD v1.1 official definitions, re-implemented)
# ════════════════════════════════════════════════════════════════════════
def normalize_answer(s):
    """Lower, remove punctuation, articles (a/an/the) and extra whitespace."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def squad_em(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def squad_f1(prediction, gold):
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_over_golds(fn, prediction, golds):
    return max(fn(prediction, g) for g in golds)


def evaluate_predictions(predictions, examples):
    """predictions: {id: text}. examples: iterable of SQuAD dicts. Returns EM/F1 in %."""
    em = f1 = 0.0
    n = 0
    for ex in examples:
        pred = predictions.get(ex['id'], '')
        golds = ex['answers']['text']
        em += max_over_golds(squad_em, pred, golds)
        f1 += max_over_golds(squad_f1, pred, golds)
        n += 1
    return {'exact_match': 100.0 * em / n, 'f1': 100.0 * f1 / n, 'n': n}


# ════════════════════════════════════════════════════════════════════════
# 2. Character span -> token span alignment
# ════════════════════════════════════════════════════════════════════════
def char_to_token_span(offsets, sequence_ids, answer_start, answer_end):
    """Map a character span [answer_start, answer_end) of the context to token indices.

    offsets      : list of (char_start, char_end) per token (exclusive end)
    sequence_ids : list of None / 0 (question) / 1 (context) per token
    Returns (start_token, end_token, status) where status is one of
      'ok'           : answer fully inside this window
      'outside'      : answer (partially) outside this window -> label (0, 0) = [CLS]
    Handles answers that begin/end in the middle of a sub-word token: the token
    containing the boundary character is used (start <= a_s < end ; start < a_e <= end).
    """
    ctx_idx = [i for i, s in enumerate(sequence_ids) if s == 1]
    ctx_first, ctx_last = ctx_idx[0], ctx_idx[-1]
    # Answer not fully inside the context part of this window -> no-answer label
    if offsets[ctx_first][0] > answer_start or offsets[ctx_last][1] < answer_end:
        return 0, 0, 'outside'

    start_token = end_token = None
    for token_idx in range(ctx_first, ctx_last + 1):
        start, end = offsets[token_idx]
        if start <= answer_start < end and start_token is None:
            start_token = token_idx
        if start < answer_end <= end:
            end_token = token_idx
    # Answer may start on whitespace between tokens (e.g. leading space in gold):
    # fall back to the first token starting after answer_start / last ending before answer_end
    if start_token is None:
        start_token = next(i for i in range(ctx_first, ctx_last + 1) if offsets[i][0] >= answer_start)
    if end_token is None:
        end_token = next(i for i in range(ctx_last, ctx_first - 1, -1) if offsets[i][1] <= answer_end)
    return start_token, end_token, 'ok'


# ════════════════════════════════════════════════════════════════════════
# 3. Feature preparation (sliding windows)
# ════════════════════════════════════════════════════════════════════════
def tokenize_windows(tokenizer, questions, contexts, max_length, stride):
    questions = [q.lstrip() for q in questions]   # some SQuAD questions have leading spaces
    return tokenizer(
        questions,
        contexts,
        truncation='only_second',
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )


def prepare_train_features(examples, tokenizer, max_length, stride):
    """Batched map fn: one example -> one or more windows with start/end token labels."""
    enc = tokenize_windows(tokenizer, examples['question'], examples['context'], max_length, stride)
    sample_map = enc.pop('overflow_to_sample_mapping')
    offset_mapping = enc.pop('offset_mapping')
    enc['start_positions'], enc['end_positions'], enc['answer_in_window'] = [], [], []
    for i, offsets in enumerate(offset_mapping):
        ex_idx = sample_map[i]
        ans = examples['answers'][ex_idx]
        a_start = ans['answer_start'][0]
        a_end = a_start + len(ans['text'][0])
        s, e, status = char_to_token_span(offsets, enc.sequence_ids(i), a_start, a_end)
        enc['start_positions'].append(s)
        enc['end_positions'].append(e)
        enc['answer_in_window'].append(int(status == 'ok'))
    return enc


def prepare_eval_features(examples, tokenizer, max_length, stride):
    """Batched map fn: keep example_id and mask question/special offsets with None."""
    enc = tokenize_windows(tokenizer, examples['question'], examples['context'], max_length, stride)
    sample_map = enc.pop('overflow_to_sample_mapping')
    enc['example_id'] = []
    for i in range(len(enc['input_ids'])):
        seq_ids = enc.sequence_ids(i)
        enc['example_id'].append(examples['id'][sample_map[i]])
        enc['offset_mapping'][i] = [o if seq_ids[k] == 1 else None
                                    for k, o in enumerate(enc['offset_mapping'][i])]
    return enc


# ════════════════════════════════════════════════════════════════════════
# 4. Span decoding
# ════════════════════════════════════════════════════════════════════════
def postprocess_predictions(examples, features, all_start_logits, all_end_logits,
                            n_best=N_BEST, max_answer_tokens=MAX_ANSWER_TOKENS):
    """Aggregate window predictions -> best span per example.

    score(i, j) = start_logit[i] + end_logit[j],  i <= j,  j - i + 1 <= max_answer_tokens,
    both tokens must belong to the context. Returns {id: dict(text, start_char, end_char, score)}.
    """
    feats_per_example = collections.defaultdict(list)
    for f_idx, ex_id in enumerate(features['example_id']):
        feats_per_example[ex_id].append(f_idx)

    offsets_all = features['offset_mapping']
    results = {}
    for ex in examples:
        context = ex['context']
        best = {'text': '', 'start_char': -1, 'end_char': -1, 'score': -1e9}
        for f_idx in feats_per_example[ex['id']]:
            s_logits = all_start_logits[f_idx]
            e_logits = all_end_logits[f_idx]
            offsets = offsets_all[f_idx]
            start_idx = np.argsort(s_logits)[-1:-n_best - 1:-1]
            end_idx = np.argsort(e_logits)[-1:-n_best - 1:-1]
            for s in start_idx:
                for e in end_idx:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = float(s_logits[s] + e_logits[e])
                    if score > best['score']:
                        sc, ec = offsets[s][0], offsets[e][1]
                        best = {'text': context[sc:ec], 'start_char': int(sc), 'end_char': int(ec),
                                'score': score, 'window': int(f_idx)}
        results[ex['id']] = best
    return results


# ════════════════════════════════════════════════════════════════════════
# 5. Data splits
# ════════════════════════════════════════════════════════════════════════
def load_splits(seed=SEED, dev_size=DEV_SIZE):
    """train / dev (held-out articles from SQuAD-train) / test (= official SQuAD validation)."""
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad')
    titles = sorted(set(ds['train']['title']))
    rng = random.Random(seed)
    rng.shuffle(titles)
    dev_titles, n = set(), 0
    title_counts = collections.Counter(ds['train']['title'])
    for t in titles:
        if n >= dev_size:
            break
        dev_titles.add(t)
        n += title_counts[t]
    train = ds['train'].filter(lambda x: x['title'] not in dev_titles)
    dev = ds['train'].filter(lambda x: x['title'] in dev_titles)
    return train, dev, ds['validation']


# ════════════════════════════════════════════════════════════════════════
# 6. Model helpers
# ════════════════════════════════════════════════════════════════════════
def qa_loss(start_logits, end_logits, start_positions, end_positions):
    """L = (L_start + L_end) / 2 with softmax over sequence positions."""
    import torch.nn.functional as F
    ignored = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored)
    end_positions = end_positions.clamp(0, ignored)
    l_start = F.cross_entropy(start_logits.float(), start_positions, ignore_index=ignored)
    l_end = F.cross_entropy(end_logits.float(), end_positions, ignore_index=ignored)
    return (l_start + l_end) / 2, l_start, l_end


MODEL_INPUTS = ('input_ids', 'attention_mask', 'token_type_ids')


def predict_logits(model, features, device, batch_size=64, with_loss=False):
    """Run the model over (eval or train-format) features. Returns start/end logits (+ mean loss)."""
    import torch
    model.eval()
    cols = [c for c in MODEL_INPUTS if c in features.column_names]
    data = features.select_columns(cols + (['start_positions', 'end_positions'] if with_loss else []))
    data = data.with_format('numpy')
    starts, ends, losses = [], [], []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        inputs = {c: torch.as_tensor(batch[c], device=device) for c in cols}
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(**inputs)
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
        if with_loss:
            sp = torch.as_tensor(batch['start_positions'], device=device)
            ep = torch.as_tensor(batch['end_positions'], device=device)
            loss, _, _ = qa_loss(out.start_logits, out.end_logits, sp, ep)
            losses.append(loss.item() * len(sp))
    s, e = np.concatenate(starts), np.concatenate(ends)
    if with_loss:
        return s, e, sum(losses) / len(features)
    return s, e


class QAEvaluator:
    """Bundles an example set with its eval features (for EM/F1) and train-style features (for loss)."""

    def __init__(self, examples, tokenizer, max_length, stride, num_proc=8):
        self.examples = examples
        self.eval_features = examples.map(
            prepare_eval_features, batched=True, remove_columns=examples.column_names,
            fn_kwargs=dict(tokenizer=tokenizer, max_length=max_length, stride=stride), num_proc=num_proc)
        self.loss_features = examples.map(
            prepare_train_features, batched=True, remove_columns=examples.column_names,
            fn_kwargs=dict(tokenizer=tokenizer, max_length=max_length, stride=stride), num_proc=num_proc)

    def __call__(self, model, device, with_loss=True):
        s, e = predict_logits(model, self.eval_features, device)
        preds = postprocess_predictions(self.examples, self.eval_features, s, e)
        metrics = evaluate_predictions({k: v['text'] for k, v in preds.items()}, self.examples)
        if with_loss:
            _, _, metrics['loss'] = predict_logits(model, self.loss_features, device, with_loss=True)
        return metrics, preds


def load_qa_model(path_or_name, device=None):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForQuestionAnswering.from_pretrained(path_or_name).to(device).eval()
    return tok, model, device


def answer_question(question, context, tokenizer, model, device, max_length=384, stride=128, top_k=1):
    """Single (question, context) inference using the same windowing + decoding as evaluation."""
    from datasets import Dataset
    ex = Dataset.from_dict({'id': ['q0'], 'question': [question], 'context': [context]})
    feats = ex.map(prepare_eval_features, batched=True, remove_columns=ex.column_names,
                   fn_kwargs=dict(tokenizer=tokenizer, max_length=max_length, stride=stride))
    s, e = predict_logits(model, feats, device)
    if top_k == 1:
        return postprocess_predictions(ex, feats, s, e)['q0']
    cands = []
    for f_idx, offsets in enumerate(feats['offset_mapping']):
        for i in np.argsort(s[f_idx])[-N_BEST:]:
            for j in np.argsort(e[f_idx])[-N_BEST:]:
                if offsets[i] is None or offsets[j] is None or j < i or j - i + 1 > MAX_ANSWER_TOKENS:
                    continue
                cands.append({'text': context[offsets[i][0]:offsets[j][1]], 'start_char': offsets[i][0],
                              'end_char': offsets[j][1], 'score': float(s[f_idx][i] + e[f_idx][j])})
    cands.sort(key=lambda c: -c['score'])
    seen, out = set(), []
    for c in cands:
        if (c['start_char'], c['end_char']) not in seen:
            seen.add((c['start_char'], c['end_char']))
            out.append(c)
        if len(out) == top_k:
            break
    return out


# ════════════════════════════════════════════════════════════════════════
# 7. Training
# ════════════════════════════════════════════════════════════════════════
def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[{args.tag}] device={device} model={args.model} max_length={args.max_length} stride={args.stride}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ex, dev_ex, test_ex = load_splits(args.seed)
    if args.train_samples:
        train_ex = train_ex.shuffle(seed=args.seed).select(range(args.train_samples))
    if args.dev_samples:
        dev_ex = dev_ex.select(range(min(args.dev_samples, len(dev_ex))))
    if args.test_samples:
        test_ex = test_ex.select(range(args.test_samples))
    print(f'[{args.tag}] examples: train={len(train_ex)} dev={len(dev_ex)} test={len(test_ex)}')

    t0 = time.time()
    train_feats = train_ex.map(prepare_train_features, batched=True, remove_columns=train_ex.column_names,
                               fn_kwargs=dict(tokenizer=tokenizer, max_length=args.max_length,
                                              stride=args.stride), num_proc=8)
    in_window = np.array(train_feats['answer_in_window'])
    feat_stats = {
        'train_examples': len(train_ex), 'train_features': len(train_feats),
        'windows_per_example': len(train_feats) / len(train_ex),
        'features_answer_in_window_pct': 100.0 * in_window.mean(),
    }
    dev_eval = QAEvaluator(dev_ex, tokenizer, args.max_length, args.stride)
    test_eval = QAEvaluator(test_ex, tokenizer, args.max_length, args.stride)
    feat_stats.update({'dev_features': len(dev_eval.eval_features),
                       'test_features': len(test_eval.eval_features),
                       'preprocess_sec': time.time() - t0})
    print(f'[{args.tag}] features: {feat_stats}')

    cols = [c for c in MODEL_INPUTS if c in train_feats.column_names] + ['start_positions', 'end_positions']
    train_feats.set_format('torch', columns=cols)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(train_feats, batch_size=args.batch_size, shuffle=True, generator=g,
                        num_workers=2, drop_last=False)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    no_decay = ('bias', 'LayerNorm.weight')
    params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    history = []
    step, run_loss, run_ls, run_le, run_n = 0, 0.0, 0.0, 0.0, 0
    train_time = 0.0
    t_train = time.time()
    for epoch in range(args.epochs):
        model.train()
        for b_idx, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(**{k: batch[k] for k in cols[:-2]})
            loss, ls, le = qa_loss(out.start_logits, out.end_logits,
                                   batch['start_positions'], batch['end_positions'])
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item(); run_ls += ls.item(); run_le += le.item(); run_n += 1
            if (b_idx + 1) % args.grad_accum != 0 and b_idx + 1 != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % 50 == 0:
                print(f'[{args.tag}] ep {epoch + 1} step {step}/{total_steps} loss {run_loss / run_n:.4f}',
                      flush=True)
            if step % eval_every == 0 or step == total_steps:
                train_time += time.time() - t_train
                dev_m, _ = dev_eval(model, device)
                rec = {'step': step, 'epoch': step / steps_per_epoch,
                       'train_loss': run_loss / run_n, 'train_start_loss': run_ls / run_n,
                       'train_end_loss': run_le / run_n, 'val_loss': dev_m['loss'],
                       'val_em': dev_m['exact_match'], 'val_f1': dev_m['f1'], 'train_time_sec': train_time}
                history.append(rec)
                print(f'[{args.tag}] EVAL {json.dumps(rec)}', flush=True)
                run_loss, run_ls, run_le, run_n = 0.0, 0.0, 0.0, 0
                model.train()
                t_train = time.time()

    test_m, test_preds = test_eval(model, device)
    print(f'[{args.tag}] TEST {test_m}')
    summary = {'tag': args.tag, 'config': vars(args), **feat_stats,
               'train_time_sec': train_time, 'test': test_m, 'final_dev': history[-1],
               'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, 'test_predictions.json'), 'w') as f:
        json.dump(test_preds, f)
    if args.save_model:
        model.save_pretrained(os.path.join(out_dir, 'model'))
        tokenizer.save_pretrained(os.path.join(out_dir, 'model'))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='main')
    p.add_argument('--model', default=MODEL_NAME)
    p.add_argument('--max_length', type=int, default=384)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--grad_accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--evals_per_epoch', type=int, default=4)
    p.add_argument('--train_samples', type=int, default=0)
    p.add_argument('--dev_samples', type=int, default=0)
    p.add_argument('--test_samples', type=int, default=0)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--output_dir', default='outputs')
    p.add_argument('--save_model', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
