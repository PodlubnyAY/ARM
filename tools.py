import re
import pandas as pd
from functools import reduce
from multiprocessing import Pool
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori, fpgrowth
from tree import Tree

MIN_SUPPORT = 0.05
MIN_THRESHOLD = 0.9
FILE_FORMATS = ('txt', 'csv', 'xls', 'xlsx')
METRICS = {'confidence', 'leverage', 'conviction', 'zhangs_metric', 'lift'}


def _get_tree_rule(df, target, min_support, min_threshold, depth, width):
    t = Tree(
        df, target, 
        min_support=min_support, min_threshold=min_threshold,
        width=width, depth=depth,
    )
    t.growth()
    return t.get_rules()


def tree_rules(
    df, min_support, metric, min_threshold,
    tree_root=None, width=None, depth=None, n_proc=None, **kwargs,
):
    params = (
        (df, target, min_support, min_threshold, depth, width)
        for target in df.columns.delete(df.columns==tree_root)
    )
        
    if n_proc is None:
        rules = []
        for p in params:
            rules.append(
                _get_tree_rule(*p)
            )
    else:
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
    def wrapped(df, min_support, metric, min_threshold, **kwargs):
        data = one_hot_encode(df)
        frequent_itemsets = method(data, min_support, use_colnames=True)
        rules = association_rules(
            frequent_itemsets, 
            min_threshold=min_threshold,
            metric=metric,
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


def read_file(file, no_header, delimiter):
    header = None if no_header else 0
    file_format = file.split('.')[-1]
    if file_format not in FILE_FORMATS:
        print(
            f"Неверный формат файла: ({file_format})."
            " Необходим", *FILE_FORMATS
        )
        exit(1)
    if file_format.startswith('xls'):
        buff = pd.read_excel(file, header=header)
        if no_header:
            buff.columns = [f'X{i}' for i in range(buff.shape[1])]
        return buff
    buff = pd.read_csv(file, header=header, delimiter=delimiter)
    if no_header:
        buff.columns = [f'X{i}' for i in range(buff.shape[1])]
    return buff



def print_rules(rules, metric, save_to=None, delimiter=','):
    if not rules.shape[0]:
        print("Нет логических зависимостей с заданными ограничениями")
        return
    
    rules = rules.round(2)
    rules: pd.DataFrame = rules.sort_values(by=['support', metric], ascending=False)
    rules.reset_index(inplace=True, drop=True)
    print(rules.to_string())
    if save_to:
        if save_to.endswith('csv') or save_to.endswith('txt'):
            rules.to_csv(save_to, sep=delimiter, index=False)
        elif save_to.endswith('xlsx') or save_to.endswith('xls'):
            rules.to_excel(save_to, index=False)
        else:
            print(f"Неверный формат файла {save_to}. Необходим {FILE_FORMATS}")


def get_rules(
    method, df, min_support, metric, min_threshold, **kwargs# tree_root, n_proc=None, width=0, depth=0,
):
    methods = {
        "apriori": freq_itemset_rules(apriori), 
        "fpgrowth": freq_itemset_rules(fpgrowth),
        "tree": tree_rules,
    }
    method = methods.get(method, fpgrowth)
    base_rules = method(
        df, min_support=min_support,
        min_threshold=min_threshold,
        metric=metric,
        **kwargs,
        # n_proc=n_proc,
        # width=width,
        # depth=depth,
    )
    return base_rules
