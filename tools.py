import re
import pandas as pd
from functools import reduce
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori, fpgrowth
from tree import Tree

MIN_SUPPORT = 0.05
MIN_THRESHOLD = 0.9


def tree_rules(df, min_support, min_threshold, root=None):
    rules = []
    for target in df.columns:
        t = Tree(
            df, target, 
            min_support=min_support, min_threshold=min_threshold,
            supposed_root_attribute=root,
        )
        t.growth()
        r = t.get_rules()
        if r.shape[0]:
            rules.append(r)
        
    result = pd.concat(rules)
    result['lift'] = result['confidence'] / result['support']
    # result['leverage'] = 
    # TODO: append metrics
    return result


def one_hot_encode(df, labels=False):
    if isinstance(df, pd.DataFrame):
        transactions = [
            [f"{column}={transaction[i]}" for i, column in enumerate(df.columns)]
            for transaction in df.itertuples(index=False)
        ]
        labels_ = set(reduce(lambda l1, l2: l1 + l2, transactions))
    elif isinstance(df, pd.Series):
        transactions = [
            [f"{df.name}={transaction}"]
            for transaction in df
        ]
        labels_ = set([t[0] for t in transactions])
    else:
        raise TypeError(f"Bad type: {type(df)}")
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    if labels:
        return df, tuple(labels_)
    return df


def freq_itemset_rules(method):
    def wrapped(df, min_support, min_threshold, root=None):
        data = one_hot_encode(df)
        frequent_itemsets = method(data, min_support, use_colnames=True)
        rules = association_rules(
            frequent_itemsets, 
            min_threshold=min_threshold,
        )
        rules['support'] = rules['antecedent support']
        return rules.drop(columns=['antecedent support', 'consequent support'])
    return wrapped


def filter_rows_postprocessing(df, condition, column):
    condition = re.split(r'\s*&{1,2}\s*', condition)
    return df[df[column] == frozenset(condition)]


def parsel_and_conclusion(df, parsel, conclusion):
    conditions = {
        name: re.split(r'\s*&{1,2}\s*', condition)
        for name, condition in zip(("antecedents", "consequents"),
                                   (parsel, conclusion))
    }
    return df[(df["antecedents"] == frozenset(conditions["antecedents"])) &
              (df["consequents"] == frozenset(conditions["consequents"]))]


METRICS = {'confidence', 'leverage', 'conviction', 'zhangs_metric', 'lift'}
METHODS = {
    "apriori": freq_itemset_rules(apriori), 
    "fpgrowth": freq_itemset_rules(fpgrowth),
    "tree": tree_rules,
}
ROUTER = {
    "parsel": lambda df, args: filter_rows_postprocessing(df, args.parsel, "antecedents"),
    "conclusion": lambda df, args: filter_rows_postprocessing(df, args.conclusion, "consequents"),
    "parsel_and_conclusion": parsel_and_conclusion,
#     "factors": lambda df, _: df,
#     "all": lambda df, _: df
}
