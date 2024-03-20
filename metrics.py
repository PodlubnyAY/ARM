import numpy as np
import pandas as pd


def calc_support(s: pd.Series, v):
    return s[s == v].size / s.size


metric_dict = {
    "antecedent support": lambda _, sA, __: sA,
    "consequent support": lambda _, __, sC: sC,
    "support": lambda _, sA, __: sA,
    "confidence": lambda sAC, sA, _: sAC / sA,
    "lift": lambda sAC, sA, sC: sAC / sA / sC,
    "leverage": lambda sAC, sA, sC: sAC - sA * sC,
}


