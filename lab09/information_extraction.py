"""Information Extraction pipeline using fine-tuned BERT Token Classifier.

Converts unstructured text to structured JSON entities:
Input: 'Barack Obama visited Paris.'
Output: {
    'PERSON': ['Barack Obama'],
    'LOCATION': ['Paris']
}
"""

import json
from collections import defaultdict
from datasets import load_dataset
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ID2TAG = {i: t for i, t in enumerate(TAG_NAMES)}
TAG_MAP = {'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISCELLANEOUS'}


class InformationExtractor:
    def __init__(self, model_name='dslim/bert-base-NER', checkpoint_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        target_model = checkpoint_path if checkpoint_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        self.model = AutoModelForTokenClassification.from_pretrained(target_model)
        self.model.to(self.device)
        self.model.eval()
        self.id2tag = {int(k): v for k, v in self.model.config.id2label.items()}

    def extract_entities(self, text):
        words = text.split()
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)
            preds = outputs.logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        word_ids = enc.word_ids(batch_index=0)
        word_tags = []
        prev_wid = None
        for wid, pid in zip(word_ids, preds):
            if wid is not None and wid != prev_wid:
                word_tags.append(self.id2tag[pid])
                prev_wid = wid

        entities = defaultdict(list)
        cur_entity = []
        cur_type = None

        def add_entity(entity_tokens, entity_type):
            name = " ".join(entity_tokens).strip(".,;:!?\"'()[]")
            if name and entity_type:
                entities[TAG_MAP.get(entity_type, entity_type)].append(name)

        for w, tag in zip(words, word_tags):
            if tag.startswith('B-'):
                if cur_entity:
                    add_entity(cur_entity, cur_type)
                cur_type = tag[2:]
                cur_entity = [w]
            elif tag.startswith('I-') and cur_type == tag[2:]:
                cur_entity.append(w)
            else:
                if cur_entity:
                    add_entity(cur_entity, cur_type)
                    cur_entity = []
                    cur_type = None

        if cur_entity:
            add_entity(cur_entity, cur_type)

        return dict(entities)


def evaluate_on_benchmark(extractor, num_sentences=30):
    ds = load_dataset('lhoestq/conll2003', split='test')
    gold_seqs = []
    pred_seqs = []

    for i in range(min(num_sentences, len(ds))):
        words = ds[i]['tokens']
        gold_tags = [ID2TAG[t] for t in ds[i]['ner_tags']]
        text = " ".join(words)

        enc = extractor.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(extractor.device)

        with torch.no_grad():
            preds = extractor.model(**enc).logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        word_ids = enc.word_ids(batch_index=0)
        predicted_tags = []
        prev_wid = None
        for wid, pid in zip(word_ids, preds):
            if wid is not None and wid != prev_wid:
                predicted_tags.append(ID2TAG[pid])
                prev_wid = wid

        # pad or trim if needed
        while len(predicted_tags) < len(gold_tags):
            predicted_tags.append('O')
        predicted_tags = predicted_tags[:len(gold_tags)]

        gold_seqs.append(gold_tags)
        pred_seqs.append(predicted_tags)

    print(f"\n--- Information Extraction Benchmark ({num_sentences} sentences) ---")
    print(seq_report(gold_seqs, pred_seqs, digits=4))
    f1 = seq_f1(gold_seqs, pred_seqs)
    print(f"Benchmark Entity-Level F1: {f1:.4f}")
    return f1


def main():
    ie = InformationExtractor(model_name='bert-base-cased')

    demo_sentences = [
        "Barack Obama visited Paris yesterday.",
        "Microsoft and Google signed a partnership agreement in New York.",
        "The United Nations held an emergency summit in Geneva on Friday.",
        "Elon Musk announced new SpaceX launch facilities in California.",
        "Apple Inc reported record revenue at its Cupertino headquarters."
    ]

    print("=== Information Extraction Structured Outputs ===")
    for sent in demo_sentences:
        res = ie.extract_entities(sent)
        print(f"\nText:   '{sent}'")
        print(f"Output: {json.dumps(res, ensure_ascii=False)}")

    evaluate_on_benchmark(ie, num_sentences=30)


if __name__ == '__main__':
    main()
