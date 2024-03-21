import re
import pandas as pd
from functools import reduce
from itertools import combinations
from multiprocessing import Pool
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori, fpgrowth
from tree import Tree

MIN_SUPPORT = 0.05
MIN_THRESHOLD = 0.9


def _get_tree_rule(df, target, min_support, min_threshold, root):
    t = Tree(
        df, target, 
        min_support=min_support, min_threshold=min_threshold,
        supposed_root_attribute=root,
    )
    t.growth()
    return t.get_rules()


def tree_rules(df, min_support, min_threshold, root=None, n_proc=None):
    if root is None:
        params = []
        for col1, col2 in combinations(df.columns, 2):
            params.extend((
                (df, col1, min_support, min_threshold, col2),
                (df, col2, min_support, min_threshold, col1),
            ))
    else:
        params = (
            (df, target, min_support, min_threshold, root)
            for target, in df.columns
        )
        
    n_proc = 4 if n_proc is None else int(n_proc)
    with Pool(n_proc) as tree_pool:
        rules = tree_pool.starmap(_get_tree_rule, params)
    
    result = pd.concat(rules)
    result['lift'] = result['confidence'] / result['support']
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
    def wrapped(df, min_support, min_threshold, root=None, n_proc=None):
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
