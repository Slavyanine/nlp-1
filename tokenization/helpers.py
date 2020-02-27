import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
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


def example_tasks():
    jaccard_str = 'S1 = {}\nS2 = {}\nJaccard index = {}\nJaccard distance = {}\n'
    jaro_str = 'S1 = {}\nS2 = {}\nJaro similarity = {}\nJaro-Winkler similarity = {}\n'
    s1, s2 = 'I do not like green eggs and ham', 'I do not like them, Sam I am'
    s3, s4 = 'aaaaaaaa', 'aaabbbbbbaa'
    s5, s6 = 'CRATE', 'TRACE'
    s7, s8 = ['M', 'A', 'R', 'H', 'T', 'A'], ['M', 'A', 'R', 'T', 'H', 'A']
    s9, s10 = 'DICKSONX', 'DIXON'
    print(jaccard_str.format(s1, s2, get_jaccard_index(s1, s2, is_sentence=True),
                             get_jaccard_distance(s1, s2, is_sentence=True)))
    print(jaccard_str.format(s3, s4, get_jaccard_index(s3, s4, is_word=True),
                             get_jaccard_distance(s3, s4, is_sentence=False)))
    print(jaccard_str.format(s5, s6, get_jaccard_index(s5, s6, is_word=True),
                             get_jaccard_distance(s5, s6, is_sentence=False)))
    print(jaro_str.format(s7, s8, get_jaro_similarity(s7, s8), get_jaro_winkler_similarity(s7, s8)))
    print(jaro_str.format(s9, s10, get_jaro_similarity(s9, s10), get_jaro_winkler_similarity(s9, s10)))
    print(jaro_str.format(s5, s6, get_jaro_similarity(s5, s6), get_jaro_winkler_similarity(s5, s6)))


def example_comparing():
    text = "Natural language processing (NLP) is a field " + \
           "of computer science, artificial intelligence " + \
           "and computational linguistics concerned with " + \
           "the interactions between computers and human " + \
           "(natural) languages, and, in particular, " + \
           "concerned with programming computers to " + \
           "fruitfully process large natural language " + \
           "corpora. Challenges in natural language " + \
           "processing frequently involve natural " + \
           "language understanding, natural language" + \
           "generation frequently from formal, machine" + \
           "-readable logical forms), connecting language " + \
           "and machine perception, managing human-" + \
           "computer dialog systems, or some combination."
    df = compare_methods(text)
    jaccard = df.jaccard_distance
    jaro = df.jaro_similarity
    jaro_winkler = df.jaro_winkler_similarity
    sns.distplot(jaccard)
    plt.show()
    sns.distplot(jaro)
    plt.show()
    sns.distplot(jaro_winkler)
    plt.show()
    sns.distplot(jaro)
    sns.distplot(jaccard)
    sns.distplot(jaro_winkler)
    plt.show()

    df.index = np.arange(0, len(df))
    print(df)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(df.drop(['a', 'b'], axis=1))
    plot_dendrogram(model, truncate_mode='level')
    plt.show()

    df = pd.melt(df,
                 id_vars=['a', 'b'],
                 value_vars=['jaccard_index', 'jaccard_distance', 'jaro_similarity', 'jaro_winkler_similarity'],
                 var_name='method',
                 value_name='score')
    lp = sns.lineplot(x='a', y='score', hue='method', data=df)
    lp.set_xticklabels(lp.get_xticklabels(), rotation=90)
    plt.show()
