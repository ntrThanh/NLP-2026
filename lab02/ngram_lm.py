import math
from collections import Counter


def build_vocabulary(corpus):
    vocabulary = set()
    for text in corpus:
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        for token in tokens:
            vocabulary.add(token)
    return sorted(list(vocabulary))


def count_ngrams(corpus, n):
    counts = Counter()
    for text in corpus:
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text

        if n == 1:
            for token in tokens:
                counts[token] += 1
        elif len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                counts[ngram] += 1
    return counts


def train_unigram(corpus):
    counts = count_ngrams(corpus, 1)
    total_tokens = sum(counts.values())
    probs = {}
    if total_tokens > 0:
        for word, count in counts.items():
            probs[word] = count / total_tokens
    return probs, counts, total_tokens


def train_bigram(corpus):
    unigram_counts = count_ngrams(corpus, 1)
    bigram_counts = count_ngrams(corpus, 2)
    probs = {}
    for (w1, w2), count in bigram_counts.items():
        denom = unigram_counts.get(w1, 0)
        if denom > 0:
            probs[(w1, w2)] = count / denom
    return probs, bigram_counts, unigram_counts


def train_trigram(corpus):
    bigram_counts = count_ngrams(corpus, 2)
    trigram_counts = count_ngrams(corpus, 3)
    probs = {}
    for (w1, w2, w3), count in trigram_counts.items():
        denom = bigram_counts.get((w1, w2), 0)
        if denom > 0:
            probs[(w1, w2, w3)] = count / denom
    return probs, trigram_counts, bigram_counts


def probability(context, word, counts, context_counts, vocab_size, smoothing=None):
    if isinstance(context, str):
        context = tuple(context.split()) if context else ()
    elif context is None:
        context = ()
    else:
        context = tuple(context)

    if len(context) == 0:
        count_w = counts.get(word, 0)
        total = context_counts if isinstance(context_counts, (int, float)) else sum(counts.values())
        if smoothing == "laplace":
            return (count_w + 1) / (total + vocab_size) if (total + vocab_size) > 0 else 0.0
        return count_w / total if total > 0 else 0.0

    ngram = context + (word,)
    count_ngram = counts.get(ngram, 0)
    count_context = context_counts.get(context, 0) if isinstance(context_counts, dict) else context_counts.get(context[0] if len(context) == 1 else context, 0)

    if smoothing == "laplace":
        denom = count_context + vocab_size
        return (count_ngram + 1) / denom if denom > 0 else 0.0

    return count_ngram / count_context if count_context > 0 else 0.0


def sentence_probability(sentence, model):
    return model.sentence_probability(sentence)


def sentence_log_probability(sentence, model):
    return model.sentence_log_probability(sentence)


class NGramLanguageModel:
    def __init__(self, n, smoothing=None):
        self.n = n
        self.smoothing = smoothing
        self.vocabulary = set()
        self.vocab_size = 0
        self.total_tokens = 0
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def fit(self, corpus):
        tokenized_corpus = []
        for text in corpus:
            if isinstance(text, str):
                tokens = text.split()
            else:
                tokens = list(text)
            tokenized_corpus.append(tokens)

        self.vocabulary = set(build_vocabulary(tokenized_corpus))
        self.vocab_size = len(self.vocabulary)

        if self.n == 1:
            self.ngram_counts = count_ngrams(tokenized_corpus, 1)
            self.total_tokens = sum(self.ngram_counts.values())
            self.context_counts = self.total_tokens
        elif self.n == 2:
            self.ngram_counts = count_ngrams(tokenized_corpus, 2)
            self.context_counts = count_ngrams(tokenized_corpus, 1)
            self.total_tokens = sum(self.context_counts.values())
        elif self.n == 3:
            self.ngram_counts = count_ngrams(tokenized_corpus, 3)
            self.context_counts = count_ngrams(tokenized_corpus, 2)
            unigram_counts = count_ngrams(tokenized_corpus, 1)
            self.total_tokens = sum(unigram_counts.values())
            self.unigram_counts = unigram_counts

    def probability(self, context, word):
        if self.n == 1:
            count_w = self.ngram_counts.get(word, 0)
            if self.smoothing == "laplace":
                return (count_w + 1) / (self.total_tokens + self.vocab_size) if (self.total_tokens + self.vocab_size) > 0 else 0.0
            return count_w / self.total_tokens if self.total_tokens > 0 else 0.0

        if isinstance(context, str):
            context = tuple(context.split()) if context else ()
        elif context is None:
            context = ()
        else:
            context = tuple(context)

        expected_context_len = self.n - 1
        if len(context) > expected_context_len:
            context = context[-expected_context_len:]
        elif len(context) < expected_context_len:
            if self.n == 3 and len(context) == 1:
                c_bi = self.context_counts.get((context[0], word), 0)
                c_uni = getattr(self, "unigram_counts", {}).get(context[0], 0)
                if self.smoothing == "laplace":
                    return (c_bi + 1) / (c_uni + self.vocab_size) if (c_uni + self.vocab_size) > 0 else 0.0
                return c_bi / c_uni if c_uni > 0 else 0.0
            elif len(context) == 0:
                count_w = self.context_counts.get(word, 0) if self.n == 2 else getattr(self, "unigram_counts", {}).get(word, 0)
                if self.smoothing == "laplace":
                    return (count_w + 1) / (self.total_tokens + self.vocab_size) if (self.total_tokens + self.vocab_size) > 0 else 0.0
                return count_w / self.total_tokens if self.total_tokens > 0 else 0.0

        ngram = context + (word,)
        count_ngram = self.ngram_counts.get(ngram, 0)
        count_ctx = self.context_counts.get(context[0] if len(context) == 1 else context, 0)

        if self.smoothing == "laplace":
            return (count_ngram + 1) / (count_ctx + self.vocab_size) if (count_ctx + self.vocab_size) > 0 else 0.0

        return count_ngram / count_ctx if count_ctx > 0 else 0.0

    def sentence_probability(self, sentence):
        if isinstance(sentence, str):
            tokens = sentence.split()
        else:
            tokens = list(sentence)

        if not tokens:
            return 0.0

        prob = 1.0
        for i in range(len(tokens)):
            if self.n == 1:
                p = self.probability((), tokens[i])
            elif self.n == 2:
                if i == 0:
                    p = self.probability((), tokens[i])
                else:
                    p = self.probability(tokens[i - 1], tokens[i])
            elif self.n == 3:
                if i == 0:
                    p = self.probability((), tokens[i])
                elif i == 1:
                    p = self.probability((tokens[i - 1],), tokens[i])
                else:
                    p = self.probability((tokens[i - 2], tokens[i - 1]), tokens[i])

            prob *= p
            if prob == 0.0:
                return 0.0

        return prob

    def sentence_log_probability(self, sentence):
        if isinstance(sentence, str):
            tokens = sentence.split()
        else:
            tokens = list(sentence)

        if not tokens:
            return -math.inf

        log_prob = 0.0
        for i in range(len(tokens)):
            if self.n == 1:
                p = self.probability((), tokens[i])
            elif self.n == 2:
                if i == 0:
                    p = self.probability((), tokens[i])
                else:
                    p = self.probability(tokens[i - 1], tokens[i])
            elif self.n == 3:
                if i == 0:
                    p = self.probability((), tokens[i])
                elif i == 1:
                    p = self.probability((tokens[i - 1],), tokens[i])
                else:
                    p = self.probability((tokens[i - 2], tokens[i - 1]), tokens[i])

            if p <= 0.0:
                return -math.inf
            log_prob += math.log(p)

        return log_prob

    def next_word_distribution(self, context):
        distribution = {}
        for word in self.vocabulary:
            p = self.probability(context, word)
            if p > 0:
                distribution[word] = p
        return sorted(distribution.items(), key=lambda item: item[1], reverse=True)

    def continuation_log_probability(self, context, continuation):
        if isinstance(context, str):
            c_tokens = context.split()
        else:
            c_tokens = list(context)

        if isinstance(continuation, str):
            a_tokens = continuation.split()
        else:
            a_tokens = list(continuation)

        if not a_tokens:
            return 0.0

        full_tokens = c_tokens + a_tokens
        log_prob = 0.0

        for i in range(len(c_tokens), len(full_tokens)):
            target = full_tokens[i]
            ctx = tuple(full_tokens[max(0, i - (self.n - 1)) : i])
            p = self.probability(ctx, target)
            if p <= 0.0:
                return -math.inf
            log_prob += math.log(p)

        return log_prob

    def continuation_probability(self, context, continuation):
        lp = self.continuation_log_probability(context, continuation)
        if lp == -math.inf:
            return 0.0
        return math.exp(lp)

    def rank_continuations(self, context, candidates):
        results = []
        for cand in candidates:
            cand_tokens = cand.split() if isinstance(cand, str) else list(cand)
            lp = self.continuation_log_probability(context, cand_tokens)
            prob = math.exp(lp) if lp != -math.inf else 0.0
            k = len(cand_tokens)
            ppl = math.exp(-lp / k) if (lp != -math.inf and k > 0) else float("inf")
            results.append({
                "candidate": cand if isinstance(cand, str) else " ".join(cand),
                "log_prob": lp,
                "probability": prob,
                "perplexity": ppl,
                "length": k
            })
        return sorted(results, key=lambda x: x["probability"], reverse=True)


if __name__ == "__main__":
    toy_corpus = [
        "the cat eats fish",
        "the cat likes fish",
        "the dog eats meat",
    ]

    vocab = build_vocabulary(toy_corpus)
    print("Vocabulary:", vocab)

    uni_lm = NGramLanguageModel(n=1)
    uni_lm.fit(toy_corpus)
    print("P(the):", uni_lm.probability((), "the"))

    bi_lm = NGramLanguageModel(n=2)
    bi_lm.fit(toy_corpus)
    print("P(cat|the):", bi_lm.probability("the", "cat"))
    print("P(dog|the):", bi_lm.probability("the", "dog"))

    tri_lm = NGramLanguageModel(n=3)
    tri_lm.fit(toy_corpus)
    print("P(eats|the cat):", tri_lm.probability(("the", "cat"), "eats"))

    test_sent = "the cat eats fish"
    print("Sentence prob (bi MLE):", bi_lm.sentence_probability(test_sent))
    print("Sentence log prob (bi MLE):", bi_lm.sentence_log_probability(test_sent))

    bi_lm_laplace = NGramLanguageModel(n=2, smoothing="laplace")
    bi_lm_laplace.fit(toy_corpus)
    print("P_laplace(cat|the):", bi_lm_laplace.probability("the", "cat"))
    print("P_laplace(eats|the):", bi_lm_laplace.probability("the", "eats"))
    print("Next words after 'the':", bi_lm.next_word_distribution("the")[:3])
