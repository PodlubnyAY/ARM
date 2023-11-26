import argparse
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

METRICS = {'confidence', 'leverage', 'conviction', 'zhangs_metric', 'lift'}
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--method', type=str, default='fpgrowth')
parser.add_argument('--min_support', type=float, default=0.5)
parser.add_argument('--min_threshold', type=float, default=0.7)
parser.add_argument('--metric', type=str, default='confidence', choices=METRICS)
parser.add_argument('-v', '--verbose', action='store_true')


def main(args):
    df = pd.read_excel(args.input)
    transactions = [
        [f"{column}={transaction[i]}" for i, column in enumerate(df.columns)]
        for transaction in df.itertuples(index=False)
    ]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = fpgrowth(df, min_support=args.min_support, use_colnames=True)
    rules: pd.DataFrame = association_rules(frequent_itemsets, metric=args.metric, min_threshold=args.min_threshold)
    if not args.verbose:
        columns_dropped = list(METRICS - {args.metric})
        rules = rules.drop(columns_dropped, axis=1)
    rules = rules.round(2)
    print(rules.to_string(index=False))


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
