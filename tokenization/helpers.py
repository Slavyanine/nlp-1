import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import dendrogram

from text_mining.tokenization.string_metrics import get_jaccard_index, get_jaccard_distance, get_jaro_similarity, \
    get_jaro_winkler_similarity
from text_mining.tokenization.tokenizer import _word_tokenization


def _get_lower_case(_a, _b):
    return [i.lower() for i in _a], [i.lower() for i in _b]


def plot_dendrogram(_model, **kwargs):
    counts = np.zeros(_model.children_.shape[0])
    n_samples = len(_model.labels_)
    for i, merge in enumerate(_model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([_model.children_, _model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def compare_methods(_text):
    words = _word_tokenization(_text)
    stop_words = stopwords.words("english")
    without_stop_words = [word for word in words if word not in stop_words]
    pairs = [(i, (i + 1) % len(without_stop_words)) for i in range(len(without_stop_words))]
    scores_data = pd.DataFrame()
    for word_pair in pairs:
        a = without_stop_words[word_pair[0]]
        b = without_stop_words[word_pair[1]]
        jaccard_index = get_jaccard_index(a, b, is_word=True)
        jaccard_distance = get_jaccard_distance(a, b, is_sentence=False)
        jaro_similarity = get_jaro_similarity(a, b)
        jaro_winkler_similarity = get_jaro_winkler_similarity(a, b)
        scores_data = scores_data.append(pd.DataFrame({'a': a,
                                                       'b': b,
                                                       'jaccard_index': jaccard_index,
                                                       'jaccard_distance': jaccard_distance,
                                                       'jaro_similarity': jaro_similarity,
                                                       'jaro_winkler_similarity': jaro_winkler_similarity}, index=[0]))
    return scores_data

