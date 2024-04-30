import re
import pandas as pd
from typing import List

import tools
import metrics
from argparser import ArgParser

parser = ArgParser()


@parser.add
def overview(input_file:str, no_header=False, delimiter=',', n=5, verbose=False):
    """Обзор набора данных"""
    data = tools.read_file(input_file, no_header=no_header, delimiter=delimiter)
    print(data.head(n))
    if verbose:
        print(data.describe(include='all'))


@parser.add
def estimate(
    input_file:str,
    parsel:str,
    conclusion:str,
    no_header=0,
    delimiter=',',
    verbose=False,
    metric="confidence",
    save_to=None,
):
    """Оценка заданных посылки и заключения"""
    pattern = r"[\w\d_-]+\s*(={1,2})\s*(?:\d+|'[\w\d_-]+')([\s,]&\s?)?"
    data = tools.read_file(input_file, no_header, delimiter)
    if match := re.match(pattern, parsel):
        if match.group(1) != "==":
            parsel = parsel.replace(match.group(1), "==")
    else:
        print(f"Неверно введена посылка: {parsel}")
        return
    if match := re.match(pattern, conclusion):
        if match.group(1) != "==":
            conclusion = conclusion.replace(match.group(1), "==")
    else:
        print(f"Неверно введено заключение: {conclusion}")
        return

    args = (
        data.query(f"{parsel} & {conclusion}").shape[0] / data.shape[0], 
        data.query(parsel).shape[0] / data.shape[0], 
        data.query(conclusion).shape[0] / data.shape[0], 
    )
    base_dict = {
        "antecedents": [parsel], 
        "consequents": [conclusion], 
    }
    base_dict |= {name : [m(*args)] for name, m in metrics.metric_dict.items()}
    rule = pd.DataFrame(base_dict)
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rule.drop(columns=columns_dropped, errors='ignore', inplace=True)
    tools.print_rules(rule, metric, save_to=save_to, delimiter=delimiter)


@parser.add
def all_rules(
    input_file:str,
    no_header=False,
    delimiter=',',
    min_support=tools.MIN_SUPPORT, 
    min_threshold=tools.MIN_THRESHOLD, 
    method="fpgrowth",
    verbose=False, 
    metric="confidence", 
    save_to=None, 
    n_proc=None,
    width=0,
    depth=0,
):
    """Поиск всех возможных ЛЗ"""
    data = tools.read_file(input_file, no_header, delimiter)
    rules = tools.get_rules(
        method, data, min_support, metric, min_threshold, width=width, depth=depth,
        n_proc=n_proc, 
    )
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    tools.print_rules(rules, metric, save_to=save_to, delimiter=delimiter)


@parser.add
def target(
    input_file:str,
    target:str,
    is_premise=False,
    min_support=tools.MIN_SUPPORT, 
    min_threshold=tools.MIN_THRESHOLD, 
    method="fpgrowth",
    verbose=False, 
    metric="confidence", 
    save_to=None, 
    n_proc=None,
    width=0,
    depth=0,
    no_header=False,
    delimiter=',',
):
    """Поиск ЛЗ с заданным целевым фактором"""
    data = tools.read_file(input_file, no_header, delimiter)
    rules: pd.DataFrame = tools.get_rules(
        method, data, min_support, metric, min_threshold, width=width, depth=depth,
        n_proc=n_proc, 
    )
    filter_column = 'antecedents' if is_premise else 'consequents'
    mask = rules[filter_column].astype(str).str.contains(
        rf"\{{\'{target}=\w+\'}}",  # ugly hack
        regex=True,
    )
    rules = rules[mask]
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    tools.print_rules(rules, metric, save_to=save_to, delimiter=delimiter)


@parser.add
def term(
    input_file:str, 
    term:str,
    is_premise=False,
    min_support=tools.MIN_SUPPORT, 
    min_threshold=tools.MIN_THRESHOLD, 
    method="fpgrowth", 
    verbose=False, 
    metric="confidence", 
    save_to=None,
    n_proc=None,
    width=0,
    depth=0,
    no_header=0,
    delimiter=',',
):
    """Поиск заключения по заданной посылке"""
    data = tools.read_file(input_file, no_header, delimiter)
    rules = tools.get_rules(
        method, data, min_support, metric, min_threshold, n_proc=n_proc,
        width=width, depth=depth,
    )
    filter_column = 'antecedents' if is_premise else 'consequents'
    rules = tools.filter_rows_postprocessing(rules, term, filter_column)
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    tools.print_rules(rules, metric, save_to=save_to, delimiter=delimiter)


@parser.add
def factor(
    input_file:str,
    factors:List[str],
    no_header=0,
    delimiter=',',
    min_support=tools.MIN_SUPPORT,
    min_threshold=tools.MIN_THRESHOLD,
    method="fpgrowth",
    verbose=False,
    metric="confidence",
    save_to=None,
    n_proc=None,
    width=0,
    depth=0,
):
    """Поиск ЛЗ для заданных атрибутов"""
    data = tools.read_file(input_file, no_header, delimiter)
    if isinstance(factors, str):
        factors = factors.split()
        
    data = data[factors]
    rules = tools.get_rules(
        method, data, min_support, metric, min_threshold, n_proc=n_proc,
        width=width, depth=depth,
    )
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    tools.print_rules(rules, metric, save_to=save_to, delimiter=delimiter)


def main():
    parser.run()


if __name__ == '__main__':
    main()
