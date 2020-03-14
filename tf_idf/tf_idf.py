import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def _get_term_frequency(word, document):
    return float(sum([sentence.count(word) for sentence in document])) / sum([len(sentence) for sentence in document])


def _get_inverse_document_frequency(word, corpus):
    count = 0
    for doc in corpus:
        is_in_doc = False
        for sentence in doc:
            if word in sentence:
                is_in_doc = True
                break
        if is_in_doc:
            count += 1.0
    return np.log(len(corpus) / count)


def get_tfidf(word, document, corpus):
    tf = _get_term_frequency(word, document)
    idf = _get_inverse_document_frequency(word, corpus)
    return tf * idf


def get_custom_dataframe_tf_idf(_corpus):
    words_frequency, i = defaultdict(list), 0
    for document in _corpus:
        for sentence in document:
            for word in list(set(sentence)):
                tf_idf = get_tfidf(word, document, _corpus)
                words_frequency[word].append({i: tf_idf})
        i += 1
    df = pd.DataFrame()
    for i in words_frequency.items():
        series = pd.Series(np.pad([], (0, len(_corpus))))
        for j in i[1]:
            series[list(j)[0]] = list(j.values())[0]
        df[i[0]] = series
    return df.reindex(sorted(df.columns), axis=1)


def get_skilearn_dataframe_tf_idf(_corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tokens = [' '.join([' '.join(sentence) for sentence in doc]) for doc in _corpus]
    values = tfidf_vectorizer.fit_transform(tokens)
    feature_names = tfidf_vectorizer.get_feature_names()
    return pd.DataFrame(values.toarray(), columns=feature_names)
