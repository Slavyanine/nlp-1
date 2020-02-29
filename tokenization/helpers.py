def get_lower_case(_a, _b):
    return [i.lower() for i in _a], [i.lower() for i in _b]


def get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches):
    if j >= len(_b):
        j = len(_b) - 1
    j_count += 1
    if _a[i] == _b[j] and i != j and i - j < _scan_window:
        m += 1
        t_matches.append([i, j])
    return j, j_count, m


def get_scan_window(_a, _b):
    return round(max(len(_a), len(_b)) / 2) - 1


def get_prefix_length(_a, _b):
    _a, _b = get_lower_case(_a, _b)
    count = 0
    recommended_value = 4
    for i in range(len(_a)):
        if i < len(_b) and _a[i] == _b[i]:
            count += 1
    return min(count, recommended_value)
