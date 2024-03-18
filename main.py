import re
import argparse
import pandas as pd
from functools import reduce
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori, fpgrowth
from tree import Tree


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


METRICS = {'confidence', 'leverage', 'conviction', 'zhangs_metric', 'lift'}
METHODS = {
    "apriori": freq_itemset_rules(apriori), 
    "fpgrowth": freq_itemset_rules(fpgrowth),
    "tree": tree_rules,
}


def set_args(subparser, name):
    subparser.add_argument('input', type=str, help="Файл с данными для поиска зависимостей")
    subparser.add_argument(
        '-m', '--method', type=str, default='fpgrowth', choices=METHODS,
        help="Выбор метода поиска зависимостей (fpgrowth по умолчанию)"
    )
    subparser.add_argument(
        '-r', type=str, default=None, 
        help="Предполагаемый корневой атрибут (только для метода tree)",
    )
    subparser.add_argument('-v', '--verbose', action='store_true', help="Рассчитать все метрики")
    subparser.add_argument(
        '--metric', type=str, default='confidence', choices=METRICS,
        help="Выбрать метрику (confidence по умолчанию)",
    )
    if name == 'parsel_and_conclusion':
        subparser.add_argument('--min_support', type=float, default=0.1)
        subparser.add_argument('--min_threshold', type=float, default=0.1)
    else:
        subparser.add_argument(
            '--min_support', type=float, default=0.05,
            help="Указать требуемую нижниюю границу значения поддержки (0.05 по умолчанию)",
        )
        subparser.add_argument(
            '--min_threshold', type=float, default=0.7,
            help="Указать требуемую нижниюю границу значения выбранной метрики (0.7 по умолчанию)",
        )


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title="Subcommands", dest="command")

parser_parsel = subparsers.add_parser("parsel", help='Найти все заключения по заданной посылке')
set_args(parser_parsel, "parsel")
parser_parsel.add_argument("parsel", type=str, help="Посылка")

parser_conclusion = subparsers.add_parser("conclusion", help='Найти все посылки по заданноому заключению')
set_args(parser_conclusion, "conclusion")
parser_conclusion.add_argument("conclusion", type=str)

parser_parsel_and_conclusion = subparsers.add_parser(
    "parsel_and_conclusion", help='Оценить заданные посылку и заключение'
)
set_args(parser_parsel_and_conclusion, "parsel_and_conclusion")
parser_parsel_and_conclusion.add_argument("parsel", type=str)
parser_parsel_and_conclusion.add_argument("conclusion", type=str)

parser_factors = subparsers.add_parser(
    "factors", 
    help='Найти все логические зависимости для заданных свойств',
)
set_args(parser_factors, "factors")
parser_factors.add_argument("factors", type=str, nargs='+', help="Факторы для поиска логических зависимостей")

parser_all = subparsers.add_parser("all", help='Найти все возможные логические зависимости')
set_args(parser_all, "all")


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


def main(args):
    try:
        df = pd.read_excel(args.input)
    except FileNotFoundError:
        print(f"{args.input} не найден")
        return

    if args.command == "factors":
        df = filter_columns(df, args.factors)
    
    method = METHODS.get(args.method, fpgrowth)
    base_rules = method(
        df, min_support=args.min_support,
        min_threshold=args.min_threshold,
        root=args.r,
    )
    # print(frequent_itemsets);exit()
    rules: pd.DataFrame = ROUTER[args.command](base_rules, args)
    if not args.verbose:
        columns_dropped = list(METRICS - {args.metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)

    rules = rules.round(2)
    if rules.shape[0]:
        rules = rules.sort_values(by=['support', args.metric], ascending=False)
        print(rules.to_string(index=False))
    else:
        print("Нет логических зависимостей с заданными ограничениями")


if __name__ == '__main__':
    args = parser.parse_args(
        'all --method tree -r Feature4 --min_threshold 0.69 validate_input.xlsx'.split()
    )
    main(args)
