import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from sklearn import tree 
# from sklearn.cluster import dbscan
from functools import reduce

METRICS = {'confidence', 'leverage', 'conviction', 'zhangs_metric', 'lift'}
METHODS = {"apriori": apriori, "fpgrowth": fpgrowth, "tree": ...}


def set_args(subparser, name):
    subparser.add_argument('input', type=str, help="Файл с данными для поиска зависимостей")
    subparser.add_argument('-m', '--method', type=str, default='fpgrowth', choices=METHODS,
                           help="Выбор метода поиска зависимостей (fpgrowth по умолчанию)")
    subparser.add_argument('-v', '--verbose', action='store_true', help="Рассчитать все метрики")
    subparser.add_argument('--metric', type=str, default='confidence', choices=METRICS,
                           help="Выбрать метрику (confidence по умолчанию)")
    if name == 'parsel_and_conclusion':
        subparser.add_argument('--min_support', type=float, default=0.1)
        subparser.add_argument('--min_threshold', type=float, default=0.1)
    else:
        subparser.add_argument('--min_support', type=float, default=0.5,
                               help="Указать требуемую нижниюю границу значения поддержки (0.5 по умолчанию)")
        subparser.add_argument('--min_threshold', type=float, default=0.7,
                               help="Указать требуемую нижниюю границу значения выбранной метрики (0.7 по умолчанию)")


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title="Subcommands", dest="command")

parser_parsel = subparsers.add_parser("parsel", help='Найти все заключения по заданной посылке')
set_args(parser_parsel, "parsel")
parser_parsel.add_argument("parsel", type=str, help="Посылка")

parser_conclusion = subparsers.add_parser("conclusion", help='Найти все посылки по заданноому заключению')
set_args(parser_conclusion, "conclusion")
parser_conclusion.add_argument("conclusion", type=str)

parser_parsel_and_conclusion = subparsers.add_parser("parsel_and_conclusion", help='Оценить заданные посылку и заключение')
set_args(parser_parsel_and_conclusion, "parsel_and_conclusion")
parser_parsel_and_conclusion.add_argument("parsel", type=str)
parser_parsel_and_conclusion.add_argument("conclusion", type=str)

parser_factors = subparsers.add_parser("factors", help='Найти все логические зависимости для заданных свойств')
set_args(parser_factors, "factors")
parser_factors.add_argument("factors", type=str, nargs='+', help="Факторы для поиска логических зависимостей")

parser_all = subparsers.add_parser("all", help='Найти все возможные логические зависимости')
set_args(parser_all, "all")


def filter_rows_preprocessing(df, condition):
    condition = re.split(r'\s*&{1,2}\s*', condition)
    for p in condition:
        p.strip()
        column, value = re.split(r"\s*=\s*", p)
        if value.isdigit():
            value = int(value)
        df = df[df[column] == value]

    return df


def filter_columns(df, columns):
    all_columns = set(df.columns)
    filter_columns = all_columns - set(columns)
    return df.drop(list(filter_columns), axis=1)


def filter_rows_postprocessing(df, condition, column):
    condition = re.split(r'\s*&{1,2}\s*', condition)
    return df[df[column] == frozenset(condition)]


def parsel_and_conclusion(df, args):
    conditions = {
        name: re.split(r'\s*&{1,2}\s*', condition)
        for name, condition in zip(("antecedents", "consequents"),
                                   (args.parsel, args.conclusion))
    }
    return df[(df["antecedents"] == frozenset(conditions["antecedents"])) &
              (df["consequents"] == frozenset(conditions["consequents"]))]


ROUTER = {
    "parsel": lambda df, args: filter_rows_postprocessing(df, args.parsel, "antecedents"),
    "conclusion": lambda df, args: filter_rows_postprocessing(df, args.conclusion, "consequents"),
    "parsel_and_conclusion": parsel_and_conclusion,
    "factors": lambda df, _: df,
    "all": lambda df, _: df
}


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


def main(args):
    try:
        df = pd.read_excel(args.input)
    except FileNotFoundError:
        print(f"{args.input} не найден")
        return

    if args.command == "factors":
        df = filter_columns(df, args.factors)

    if args.method == 'tree':
        labels = list(df.columns)
        target_label = labels.pop(0)
        X, feature_names = one_hot_encode(
            df.drop([target_label], axis=1), 
            labels=True
        )
        target, target_names = one_hot_encode(df[target_label], labels=True)
        model = tree.DecisionTreeClassifier(criterion='entropy')
        model.fit(X, target)
        # import pdb; pdb.set_trace()
        # rules = tree.plot_tree(model, filled=True, class_names=target_names)
        # print(rules)
        # plt.show()
        print(target_names, feature_names)
        r = tree.export_text(
            model, 
            class_names=['a', 'b', 'c'], 
            feature_names=feature_names
        )
        print(r)
        return
    
    df = one_hot_encode(df)
    method = METHODS.get(args.method, fpgrowth)
    frequent_itemsets = method(
        df, 
        min_support=args.min_support, 
        use_colnames=True
    )
    # print(frequent_itemsets);exit()
    rules = association_rules(
        frequent_itemsets, 
        metric=args.metric, 
        min_threshold=args.min_threshold,
        # support_only=True,
    )
    rules = ROUTER[args.command](rules, args)
    if not args.verbose:
        columns_dropped = list(METRICS - {args.metric})
        rules = rules.drop(columns_dropped, axis=1)

    rules = rules.round(2)
    if len(rules):
        print(rules.to_string(index=False))
    else:
        print("Нет логических зависимостей с заданными ограничениями")


if __name__ == '__main__':
    args = parser.parse_args('all --min_threshold 0.6 --min_support 0.35 --method apriori validate_input.xlsx'.split()) #  --method tree
    main(args)
