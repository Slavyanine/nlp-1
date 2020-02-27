import re
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize


_NGRAM = 3


def _get_word_ngrams(_a, ngram=_NGRAM):
    if not _a:
        return None
    pattern = r'[^a-zA-Zа-яА-Я]?'
    return list(ngrams(re.sub(pattern, ' ', _a).split(), ngram))


def _get_sentence_ngrams(_a, ngram=_NGRAM):
    if not _a:
        return None
    pattern = r'[^a-zA-Zа-яА-Я]+'
    ng = ngrams(re.sub(pattern, ' ', _a).split(), ngram)
    return list(ng)


def _word_tokenization(_text):
    pattern = r'[^a-zA-Zа-яА-Я]+'
    return re.sub(pattern, ' ', _text).split()


def _sentence_tokenization(_text):
    return sent_tokenize(_text)
