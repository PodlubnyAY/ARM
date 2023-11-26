import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules


df = pd.read_excel('test_input.xlsx')
transactions = [
    [f"{column}={transaction[i]}" for i, column in enumerate(df.columns)]
    for transaction in df.itertuples(index=False)
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
rules: pd.DataFrame = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = rules.drop(['antecedent support',  'consequent support'], axis=1)
rules = rules.round(2)
print(rules.to_string(index=False))
