#!/usr/bin/env python3
"""Lab 10 — Heuristic baseline for extractive QA (no learning).

    Question
       ↓  keyword overlap (content words, crude stemming)
    best candidate sentence
       ↓  answer-type rules (who/when/how many/where/...) + proximity to matched keywords
    candidate span  (contiguous run of tokens that do NOT appear in the question)

Purpose: measure how far pure lexical matching gets, so we can tell whether the fine-tuned
Transformer really exploits question/context interaction.
"""
import json
import os
import re

from qa_model import evaluate_predictions, load_splits

STOPWORDS = set("""a an the of in on at to for from by with and or but is are was were be been being
has have had do does did what which who whom whose when where why how that this these those it its
as into than then there their they he she his her him we our you your i me my not no can could
would should will shall may might must about after before during over under between through also
s many much one did name called""".split())

TOKEN_RE = re.compile(r"\w+(?:[-'.]\w+)*|[^\w\s]")
SENT_RE = re.compile(r'[^.!?]+(?:[.!?]+|$)')
MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
          'october', 'november', 'december'}
NUMBER_WORDS = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                'hundred', 'thousand', 'million', 'billion', 'half', 'dozen'}


def stem(w):
    w = w.lower()
    for suf in ('ing', 'ed', 'es', 's', 'ly'):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def tokens_with_offsets(text, base=0):
    return [(m.group(), base + m.start(), base + m.end()) for m in TOKEN_RE.finditer(text)]


def sentences_with_offsets(context):
    return [(m.group(), m.start()) for m in SENT_RE.finditer(context) if m.group().strip()]


def answer_type(question):
    q = question.lower()
    if re.search(r'\bhow (many|much)\b|\bwhat (percentage|percent|number)\b', q):
        return 'NUM'
    if re.search(r'\bwhen\b|\bwhat (year|date|day|month|century|decade)\b|\bwhich year\b', q):
        return 'DATE'
    if re.search(r'\bwho\b|\bwhom\b|\bwhose\b', q):
        return 'PERSON'
    if re.search(r'\bwhere\b|\bwhat (city|country|state|place|region)\b', q):
        return 'PLACE'
    return 'OTHER'


def is_number(tok):
    return bool(re.fullmatch(r'[\d,.]+%?', tok)) or tok.lower() in NUMBER_WORDS


def type_score(span_tokens, atype):
    words = [t for t, _, _ in span_tokens]
    has_num = any(is_number(w) for w in words)
    has_cap = any(w[:1].isupper() for w in words)
    if atype == 'NUM':
        return 2.0 if has_num else -1.0
    if atype == 'DATE':
        has_date = any(re.fullmatch(r'1\d{3}|20\d{2}', w) or w.lower() in MONTHS for w in words)
        return 3.0 if has_date else (1.0 if has_num else -1.0)
    if atype in ('PERSON', 'PLACE'):
        return 2.0 if has_cap else -0.5
    return 0.5 if has_cap else 0.0


def baseline_answer(question, context, max_span=6):
    q_stems = {stem(t) for t, _, _ in tokens_with_offsets(question)
               if t.lower() not in STOPWORDS and t.isalnum()}
    # 1) candidate sentence = max keyword overlap (ties -> earlier sentence)
    best_sent, best_overlap = None, -1
    for sent, start in sentences_with_offsets(context):
        toks = tokens_with_offsets(sent, start)
        ov = len(q_stems & {stem(t) for t, _, _ in toks})
        if ov > best_overlap:
            best_sent, best_overlap = toks, ov
    if not best_sent:
        return {'text': '', 'start_char': -1, 'end_char': -1}

    # 2) candidate spans = runs of tokens not in the question, split at punctuation
    in_q = [stem(t) in q_stems for t, _, _ in best_sent]
    match_pos = [i for i, m in enumerate(in_q) if m]
    atype = answer_type(question)
    runs, cur = [], []
    for i, (t, s, e) in enumerate(best_sent):
        if in_q[i] or (not t.isalnum() and t not in ('-', "'", '%', '$')):
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)

    best, best_score = None, -1e9
    for run in runs:
        # trim stopwords at the edges, cap length
        while run and best_sent[run[0]][0].lower() in STOPWORDS:
            run = run[1:]
        while run and best_sent[run[-1]][0].lower() in STOPWORDS:
            run = run[:-1]
        if not run:
            continue
        run = run[:max_span]
        span = [best_sent[i] for i in run]
        dist = min((min(abs(i - m) for i in run) for m in match_pos), default=5)
        score = type_score(span, atype) - 0.3 * dist - 0.1 * len(run)
        if score > best_score:
            best, best_score = span, score
    if best is None:
        return {'text': '', 'start_char': -1, 'end_char': -1}
    sc, ec = best[0][1], best[-1][2]
    return {'text': context[sc:ec], 'start_char': sc, 'end_char': ec}


def run_baseline(examples):
    return {ex['id']: baseline_answer(ex['question'], ex['context']) for ex in examples}


if __name__ == '__main__':
    out_dir = os.path.join('outputs', 'baseline')
    os.makedirs(out_dir, exist_ok=True)
    _, dev, test = load_splits()
    summary = {}
    for name, split in (('dev', dev), ('test', test)):
        preds = run_baseline(split)
        m = evaluate_predictions({k: v['text'] for k, v in preds.items()}, split)
        summary[name] = m
        print(f'Baseline {name}: EM={m["exact_match"]:.2f} F1={m["f1"]:.2f} (n={m["n"]})')
        if name == 'test':
            with open(os.path.join(out_dir, 'test_predictions.json'), 'w') as f:
                json.dump(preds, f)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    demo_ctx = 'The capital of France is Paris. The French government is based in Paris.'
    for q in ('What is the capital of France?', 'Where is the French government based?'):
        print(q, '->', baseline_answer(q, demo_ctx)['text'])
