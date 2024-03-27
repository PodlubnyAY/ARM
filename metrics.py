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


def attribute_entropy(attr, df, target):
    entropy = 0
    for value in sorted(set(df[attr])):
        if not isinstance(value, (int, float, complex)):
                value = f"'{value}'"
        name = f"{attr} == {value}"
        df_m = df.query(name)
        entropy += calc_entropy(
            df_m[target]
        ) * df_m.shape[0] / df.shape[0]
    return entropy


def calc_support(parsel, data):
    return data.query(parsel).shape[0] / data.shape[0]


def calc_confidences(s: pd.Series, min_confidence):
    confidences = {}
    for value, count in s.value_counts().items():
        confidence = count / s.size
        if confidence > min_confidence:
            confidences[value] = confidence
    
    return confidences


metric_dict = {
    "support": lambda _, sA, __: sA,
    "confidence": lambda sAC, sA, _: sAC / sA,
    "lift": lambda sAC, sA, sC: sAC / sA / sC,
    "leverage": lambda sAC, sA, sC: sAC - sA * sC,
}


