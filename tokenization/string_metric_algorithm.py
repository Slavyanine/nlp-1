def get_jaccard_index(_a, _b):
    _a, _b = _get_lower_case(_a, _b)
    _a, _b = _transform_to_set(_a), _transform_to_set(_b)
    return len(_a.intersection(_b)) / len(_a.union(_b))


def _transform_to_set(_x):
    _xx = set()
    if not isinstance(_x, set):
        for item in _x:
            _xx.add(item)
    else:
        return _x
    return _xx


def _get_lower_case(_a, _b):
    return [i.lower() for i in _a], [i.lower() for i in _b]


def get_jaro_similarity(_a, _b):
    _a, _b = _get_lower_case(_a, _b)
    _scan_window = _get_scan_window(_a, _b)
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
            j, j_count, m = _get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches)
            j -= 1
        j = i + 1
        j_count = 0
        while j < len(_b) and j_count < _scan_window:
            j, j_count, m = _get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches)
            j += 1
    for i in range(0, len(t_matches), 2):
        t_matches[i + 1].reverse()
        if t_matches[i] == t_matches[i + 1]:
            t += 1
    return ((m / len(_a)) + (m / len(_b)) + ((m - t) / m)) * (1 / 3) if m != 0 else 0


def _get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches):
    if j >= len(_b):
        j = len(_b) - 1
    j_count += 1
    if _a[i] == _b[j] and i != j and i - j < _scan_window:
        m += 1
        t_matches.append([i, j])
    return j, j_count, m


def _get_scan_window(_a, _b):
    return round(max(len(_a), len(_b)) / 2) - 1


def _get_prefix_length(_a, _b):
    _a, _b = _get_lower_case(_a, _b)
    count = 0
    for i in range(len(_a)):
        if i < len(_b) and _a[i] == _b[i]:
            count += 1
    return count


def get_jaro_winkler_similarity(_a, _b, scaling=0.1):
    if scaling < 0 or scaling > 0.25:
        scaling = 0.25
    jaro_distance = get_jaro_similarity(_a, _b)
    prefix_length = _get_prefix_length(_a, _b)
    return jaro_distance + (prefix_length * scaling * (1 - jaro_distance))


print(get_jaccard_index({'a', 'B', 'c'}, ['b', 'c', 'd']))
print()
print(get_jaro_similarity(['M', 'A', 'R', 'H', 'T', 'A'], ['M', 'A', 'R', 'T', 'H', 'A']))
print(get_jaro_winkler_similarity(['M', 'A', 'R', 'H', 'T', 'A'], ['M', 'A', 'R', 'T', 'H', 'A']))
print()
print(get_jaro_similarity('DICKSONX', 'DIXON'))
print(get_jaro_winkler_similarity('DICKSONX', 'DIXON'))
print()
print(get_jaro_similarity('CRATE', 'TRACE'))
print(get_jaro_winkler_similarity('CRATE', 'TRACE'))
