# nltk.download('wordnet')
import pymorphy2
import nltk
import text_mining.helpers.helpers as hp
import text_mining.tokenization.tokenizer as tk
from nltk.stem import WordNetLemmatizer


def lemmatize_sentence(_sentence, rus=False):
    tokens = tk.custom_word_tokenize(_sentence)
    if rus:
        without_stopwords = set([word.lower() for word in tokens if word.lower() not in hp.get_stopwords(rus=True)])
        morph = pymorphy2.MorphAnalyzer()
        _lemmas = set()
        for token in without_stopwords:
            lemma = morph.parse(token)[0].normal_form
            _lemmas.add(lemma)
    else:
        without_stopwords = [word.lower() for word in tokens if word.lower() not in hp.get_stopwords()]
        lemmatizer = WordNetLemmatizer()
        _lemmas = []
        for token, tag in nltk.pos_tag(without_stopwords):
            mapped_tag = hp.tag_map(tag[0])
            lemma = lemmatizer.lemmatize(token, mapped_tag)
            _lemmas.append(lemma)
    return _lemmas
