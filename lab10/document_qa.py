#!/usr/bin/env python3
"""Lab 10 — Mini Document QA application.

    User question  →  given document (context is provided, NO retrieval)  →  fine-tuned QA model  →  answer span

The document is processed with the same sliding-window tokenization (max_length=384, stride=128) and
n-best span decoding as in evaluation, so documents longer than one window are supported.

Usage:
    python document_qa.py                                   # built-in demo document + questions
    python document_qa.py --context_file doc.txt --question "Who proposed the Transformer?"
    python document_qa.py --interactive --context_file doc.txt
"""
import argparse
import textwrap

from qa_model import answer_question, load_qa_model

DEMO_DOCUMENT = """\
The Transformer is a neural network architecture introduced by Vaswani et al. in the 2017 paper \
"Attention Is All You Need". It replaced recurrence with self-attention, which lets every token attend to \
every other token in the sequence in parallel. The original model was designed for machine translation and \
consisted of an encoder and a decoder, each built from six identical layers.

BERT, which stands for Bidirectional Encoder Representations from Transformers, was released by researchers \
at Google in 2018. BERT uses only the encoder part of the Transformer. It is pre-trained with two objectives: \
masked language modeling, in which about 15 percent of the input tokens are masked and must be predicted, and \
next sentence prediction. The base version of BERT has 12 layers, a hidden size of 768 and about 110 million \
parameters.

The Stanford Question Answering Dataset, known as SQuAD, was created at Stanford University and published in \
2016. It contains more than 100,000 questions posed by crowdworkers on a set of Wikipedia articles. The answer \
to every question is a segment of text, or span, from the corresponding reading passage. Systems are evaluated \
with two metrics: exact match and token-level F1. SQuAD 2.0, released in 2018, added over 50,000 unanswerable \
questions written to look similar to answerable ones.

To fine-tune BERT for extractive question answering, the question and the passage are packed into a single \
sequence separated by a special [SEP] token. A small linear layer on top of the encoder produces a start score \
and an end score for every token, and the predicted answer is the span with the highest combined score. Because \
the encoder accepts at most 512 tokens, long documents are split into overlapping windows, and the overlap is \
controlled by a parameter called the stride.
"""

DEMO_QUESTIONS = [
    # direct lexical match
    'Who introduced the Transformer architecture?',
    'How many parameters does the base version of BERT have?',
    'What percentage of input tokens are masked during pre-training?',
    # paraphrased (low lexical overlap with the answer sentence)
    'Which company developed BERT?',
    'What task was the first Transformer built for?',
    'Who wrote the SQuAD questions?',
    'How are QA systems on SQuAD scored?',
    # longer answers / boundaries
    'What did SQuAD 2.0 add?',
    'What determines how much neighbouring windows share?',
    # NOT answerable from the document — SQuAD v1.1 models must still return a span
    'Who won the 2018 FIFA World Cup?',
]

LEXICAL_PAIR = ('The capital of France is Paris. The French government is based in Paris.',
                ['What city is the capital of France?', 'Where is the French government based?',
                 'Which city hosts the French government?'])


def show(question, context, tok, model, device, top_k=3):
    cands = answer_question(question, context, tok, model, device, top_k=top_k)
    best = cands[0]
    print(f'\nQ: {question}')
    print(f'   → {best["text"]!r}   (score={best["score"]:.2f}, chars {best["start_char"]}–{best["end_char"]})')
    lo = max(0, best['start_char'] - 60)
    hi = min(len(context), best['end_char'] + 60)
    snippet = context[lo:best['start_char']] + '[[' + context[best['start_char']:best['end_char']] + ']]' + \
        context[best['end_char']:hi]
    print('   context: …' + ' '.join(snippet.split()) + '…')
    if len(cands) > 1:
        alts = ', '.join(f'{c["text"]!r} ({c["score"]:.1f})' for c in cands[1:])
        print(f'   alternatives: {alts}')
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_dir', default='outputs/main/model')
    p.add_argument('--context_file')
    p.add_argument('--question', action='append')
    p.add_argument('--interactive', action='store_true')
    args = p.parse_args()

    tok, model, device = load_qa_model(args.model_dir)
    context = open(args.context_file).read() if args.context_file else DEMO_DOCUMENT
    questions = args.question or ([] if args.interactive else DEMO_QUESTIONS)
    n_tokens = len(tok(context)['input_ids'])
    print(f'Model: {args.model_dir} on {device}')
    print(f'Document: {len(context)} characters, {n_tokens} tokens '
          f'({"multiple windows" if n_tokens > 384 else "single window"} at max_length=384, stride=128)')
    print(textwrap.indent(textwrap.shorten(context, 300), '   '))

    for q in questions:
        show(q, context, tok, model, device)

    if not args.context_file and not args.question and not args.interactive:
        print('\n' + '=' * 80 + '\nQuestion/context lexical matching (Experiment 4 example)')
        ctx, qs = LEXICAL_PAIR
        print('Context:', ctx)
        for q in qs:
            show(q, ctx, tok, model, device, top_k=2)

    if args.interactive:
        while True:
            try:
                q = input('\nQuestion (empty to quit): ').strip()
            except EOFError:
                break
            if not q:
                break
            show(q, context, tok, model, device)


if __name__ == '__main__':
    main()
