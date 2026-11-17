"""Baseline NER using Logistic Regression with handcrafted orthographic features."""

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1

TAG_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ID2TAG = {i: t for i, t in enumerate(TAG_NAMES)}


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


def build_feature_dict(feature_list):
    vocab = {}
    for feats in feature_list:
        for k, v in feats.items():
            tok = f"{k}={v}"
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def vectorize(feature_list, vocab):
    from scipy.sparse import dok_matrix
    X = dok_matrix((len(feature_list), len(vocab)), dtype=np.float32)
    for i, feats in enumerate(feature_list):
        for k, v in feats.items():
            tok = f"{k}={v}"
            if tok in vocab:
                X[i, vocab[tok]] = 1.0
    return X.tocsr()


def main():
    ds = load_dataset('lhoestq/conll2003')

    train_feats = []
    train_labels = []
    for ex in ds['train']:
        sf = sentence_to_features(ex['tokens'])
        train_feats.extend(sf)
        train_labels.extend([ID2TAG[t] for t in ex['ner_tags']])

    vocab = build_feature_dict(train_feats)
    X_train = vectorize(train_feats, vocab)

    le = LabelEncoder()
    le.fit(TAG_NAMES)
    y_train = le.transform(train_labels)

    clf = LogisticRegression(max_iter=150, solver='saga', C=1.0, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    test_labels_seq = []
    test_preds_seq = []
    for ex in ds['test']:
        sf = sentence_to_features(ex['tokens'])
        X_test = vectorize(sf, vocab)
        preds = le.inverse_transform(clf.predict(X_test))
        gold = [ID2TAG[t] for t in ex['ner_tags']]
        test_labels_seq.append(gold)
        test_preds_seq.append(list(preds))

    print("=== Baseline (Logistic Regression) Entity-Level Report ===")
    print(seq_report(test_labels_seq, test_preds_seq, digits=4))
    f1 = seq_f1(test_labels_seq, test_preds_seq)
    print(f"Overall Entity F1: {f1:.4f}")
    return f1


if __name__ == '__main__':
    main()
