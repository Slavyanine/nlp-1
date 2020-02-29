import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import text_mining.tokenization.string_metrics as sm
import text_mining.tokenization.tokenizer as tk
import seaborn as sns
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import dendrogram, linkage


def get_linkages(indeces, words):
    jaccard_weights = np.apply_along_axis(sm.get_jaccard_index, 0, indeces, words)
    jaro_weights = np.apply_along_axis(sm.get_jaro_similarity, 0, indeces, words)
    jaro_winkler_similarity = np.apply_along_axis(sm.get_jaro_winkler_similarity, 0, indeces, words)
    return [linkage(jaccard_weights, 'ward'), linkage(jaro_weights, 'ward'),
            linkage(jaro_winkler_similarity, 'ward')]


def plot_dendrogram(linkage_weigths, title, words):
    dendrogram(linkage_weigths,
               labels=np.array(words),
               orientation="right",
               leaf_font_size=8)
    plt.title(title)
    plt.show()


def compare_methods(_text):
    words_tk = tk.custom_word_tokenize(_text)
    without_stop_words = [word for word in words_tk if word.lower() not in stopwords.words("english")]
    indeces = np.triu_indices(len(without_stop_words), 1)
    template_title = 'Hierarchical Clustering Dendrogram - {}'
    linkages = get_linkages(indeces, without_stop_words)
    titles = [template_title.format('Jaccard'), template_title.format('Jaro'), template_title.format('Jaro-Winkler')]
    for i in range(0, 3):
        print(titles[i])
        print(linkages[i])
        plot_dendrogram(linkages[i], titles[i], without_stop_words)


def example_tasks():
    jaccard_str = 'S1 = {}\nS2 = {}\nJaccard index = {}\nJaccard distance = {}\n'
    jaro_str = 'S1 = {}\nS2 = {}\nJaro similarity = {}\nJaro-Winkler similarity = {}\n'
    s1, s2 = 'I do not like green eggs and ham', 'I do not like them, Sam I am'
    s3, s4 = 'aaaaaaaa', 'aaabbbbbbaa'
    s5, s6 = 'CRATE', 'TRACE'
    s7, s8 = 'MARHTA', 'MARTHA'
    s9, s10 = 'DICKSONX', 'DIXON'
    print(jaccard_str.format(s1, s2, sm._get_jaccard_index(s1, s2), sm._get_jaccard_distance(s1, s2)))
    print(jaccard_str.format(s3, s4, sm._get_jaccard_index(s3, s4), sm._get_jaccard_distance(s3, s4)))
    print(jaccard_str.format(s5, s6, sm._get_jaccard_index(s5, s6), sm._get_jaccard_distance(s5, s6)))
    print(jaro_str.format(s7, s8, sm._get_jaro_similarity(s7, s8), sm._get_jaro_winkler_similarity(s7, s8)))
    print(jaro_str.format(s9, s10, sm._get_jaro_similarity(s9, s10), sm._get_jaro_winkler_similarity(s9, s10)))
    print(jaro_str.format(s5, s6, sm._get_jaro_similarity(s5, s6), sm._get_jaro_winkler_similarity(s5, s6)))


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

# Example
example_tasks()

# Compare methods
compare_methods(text)