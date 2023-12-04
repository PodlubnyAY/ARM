import argparse
import re
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

METRICS = {'confidence', 'leverage', 'conviction', 'zhangs_metric', 'lift'}
METHODS = {"apriori": apriori, "fpgrowth": fpgrowth}


def set_args(subparser, name):
    subparser.add_argument('input', type=str, help="Файл с данными для поиска зависимостей")
    subparser.add_argument('--method', type=str, default='fpgrowth', choices=METHODS,
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


parser = argparse.ArgumentParser(usage='%(prog)s subcommand [options] input_file')
# parser.add_argument('-h','--help', help="Показать данное сообщение")
subparsers = parser.add_subparsers(title="Subcommands", dest="command")

parser_p = subparsers.add_parser("parsel", help='Найти все заключения по заданной посылке')
set_args(parser_p, "parsel")
parser_p.add_argument("parsel", type=str, help="Посылка")

parser_c = subparsers.add_parser("conclusion", help='Найти все посылки по заданноому заключению')
set_args(parser_c, "conclusion")
parser_c.add_argument("conclusion", type=str)

parser_p_c = subparsers.add_parser("parsel_and_conclusion", help='Оценить заданные посылку и заключение')
set_args(parser_p_c, "parsel_and_conclusion")
parser_p_c.add_argument("parsel", type=str)
parser_p_c.add_argument("conclusion", type=str)

parser_f = subparsers.add_parser("p_c_factors", help='Найти все логические зависимости для заданных свойств')
set_args(parser_f, "factors")
parser_f.add_argument("factors", type=str, nargs='+', help="Факторы для поиска логических зависимостей")

parser_a = subparsers.add_parser("all", help='Найти все возможные логические зависимости')
set_args(parser_a, "all")


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


def one_hot_encoder(df):
    transactions = [
        [f"{column}={transaction[i]}" for i, column in enumerate(df.columns)]
        for transaction in df.itertuples(index=False)
    ]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


PREPROCESSING_ROUTER = {
    "parsel": lambda df, args: filter_rows_preprocessing(df, args.parsel),
    "conclusion": lambda df, args: filter_rows_preprocessing(df, args.conclusion),
    "parsel_and_conclusion": lambda df, args: filter_rows_preprocessing(df, " & ".join([args.parsel, args.conclusion])),
    "p_c_factors": lambda df, args: filter_columns(df, args.factors),
    "all": lambda df, _: df
}


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


POSTPROCESSING_ROUTER = {
    "parsel": lambda df, args: filter_rows_postprocessing(df, args.parsel, "antecedents"),
    "conclusion": lambda df, args: filter_rows_postprocessing(df, args.conclusion, "consequents"),
    "parsel_and_conclusion": parsel_and_conclusion,
    "p_c_factors": lambda df, args: df,
    "all": lambda df, _: df
}


def main(args):
    df = pd.read_excel(args.input)
    df = PREPROCESSING_ROUTER[args.command](df, args)
    df = one_hot_encoder(df)
    method = METHODS.get(args.method, fpgrowth)
    frequent_itemsets = method(df, min_support=args.min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=args.metric, min_threshold=args.min_threshold)
    rules = POSTPROCESSING_ROUTER[args.command](rules, args)
    if not args.verbose:
        columns_dropped = list(METRICS - {args.metric})
        rules = rules.drop(columns_dropped, axis=1)
    rules = rules.round(2)
    print(rules.to_string(index=False))


if __name__ == '__main__':
    print(arguments := parser.parse_args())
    main(arguments)
