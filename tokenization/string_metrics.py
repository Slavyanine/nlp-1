import text_mining.tokenization.helpers as hp
import text_mining.tokenization.tokenizer as tk


def _get_jaccard_index(_a, _b):
    _a, _b = _a.lower(), _b.lower()
    _a, _b = tk.get_word_ngrams(_a), tk.get_word_ngrams(_b)
    _a, _b = [''.join(i) for i in _a], [''.join(i) for i in _b]
    _a, _b = set(_a), set(_b)
    intersection = len(_a & _b)
    union = len(_a | _b)
    return intersection / union if union > 0 else 1


def _get_jaccard_distance(_a, _b):
    return 1 - _get_jaccard_index(_a, _b)


def _get_jaro_similarity(_a, _b):
    _a, _b = _a.lower(), _b.lower()
    _scan_window = hp.get_scan_window(_a, _b)
    m, t, t_matches = 0, 0, []
    if len(_a) < len(_b):
        _c = _a
        _a = _b
        _b = _c
    for i in range(len(_a)):
        if i < len(_b) and _a[i] == _b[i]:
            m += 1
        j = i - 1
        j_count = 0
        while 0 < j and j_count < _scan_window:
            j, j_count, m = hp.get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches)
            j -= 1
        j = i + 1
        j_count = 0
        while j < len(_b) and j_count < _scan_window:
            j, j_count, m = hp.get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches)
            j += 1
    for i in range(0, len(t_matches), 2):
        if i + 1 < len(t_matches):
            t_matches[i + 1].reverse()
            if t_matches[i] == t_matches[i + 1]:
                t += 1
    return ((m / len(_a)) + (m / len(_b)) + ((m - t) / m)) * (1 / 3) if m != 0 else 0


def _get_jaro_winkler_similarity(_a, _b, scaling=0.1):
    if scaling is None or scaling < 0 or scaling > 0.25:
        scaling = 0.25
    jaro_distance = _get_jaro_similarity(_a, _b)
    prefix_length = hp.get_prefix_length(_a, _b)
    return jaro_distance + (prefix_length * scaling * (1 - jaro_distance))


def get_jaccard_index(coord, words):
    i, j = coord
    return _get_jaccard_index(words[i], words[j])


def get_jaccard_distance(coord, words):
    i, j = coord
    return _get_jaccard_distance(words[i], words[j])


def get_jaro_similarity(coord, words):
    i, j = coord
    return _get_jaro_similarity(words[i], words[j])


def get_jaro_winkler_similarity(coord, words, scaling=None):
    i, j = coord
    return _get_jaro_winkler_similarity(words[i], words[j], scaling)

