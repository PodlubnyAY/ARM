import pandas as pd
from math import log2
from functools import reduce


def calc_entropy(s: pd.Series):
    iterable = list(map(
        lambda x: (x / s.size) * log2(x / s.size), 
        s.value_counts(),
    ))
    return - reduce(
        lambda x, y: x + y, 
        iterable
    )


def calc_support(s: pd.Series, v):
    return s[s == v].size / s.size


def calc_confidences(s: pd.Series, min_confidence):
    confidences = {}
    for value, count in s.value_counts().items():
        confidence = count / s.size
        if confidence > min_confidence:
            confidences[value] = confidence
    
    return confidences


metric_dict = {
    "antecedent support": lambda _, sA, __: sA,
    "consequent support": lambda _, __, sC: sC,
    "support": lambda _, sA, __: sA,
    "confidence": lambda sAC, sA, _: sAC / sA,
    "lift": lambda sAC, sA, sC: sAC / sA / sC,
    "leverage": lambda sAC, sA, sC: sAC - sA * sC,
}


