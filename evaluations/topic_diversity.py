from sklearn.feature_extraction.text import CountVectorizer

def compute_TD(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD


def compute_topic_diversity(top_words, _type="TD"):
    TD = compute_TD(top_words)
    return TD