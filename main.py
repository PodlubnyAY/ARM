import re
import pandas as pd
from typing import List

import tools
import metrics
from argparser import ArgParser

parser = ArgParser()

FILE_READER = {
    'csv': pd.read_csv,
    'xlsx': pd.read_excel,
    'xls': pd.read_excel,
}

def read_file(file):
    try:
        file_format = file.split('.')[-1]
        df = FILE_READER[file_format](file)
        return df
    except KeyError as key:
        print(f"Неверный формат файла: ({key})\nНеобходим csv, xls, xlsx")
        exit(1)

def get_rules(
    method, df, min_support, min_threshold, **kwargs# tree_root, n_proc=None, width=0, depth=0,
):
    method = tools.METHODS.get(method, tools.fpgrowth)
    base_rules = method(
        df, min_support=min_support,
        min_threshold=min_threshold,
        # root=tree_root,
        **kwargs,
        # n_proc=n_proc,
        # width=width,
        # depth=depth,
    )
    return base_rules


def print_rules(rules, metric, save_to=None):
    if rules.shape[0]:
        rules = rules.round(2)
        rules = rules.sort_values(by=['support', metric], ascending=False)
        print(rules.to_string(index=False))
        if save_to:
            if save_to.endswith('csv'):
                rules.to_csv(save_to)
            elif save_to.endswith('xlsx') or save_to.endswith('xls'):
                rules.to_excel(save_to)
            else:
                print(f"Неверный формат файла {save_to}. Необходим .csv, .xls, .xlsx")
    else:
        print("Нет логических зависимостей с заданными ограничениями")


@parser.add
def print_features(input_file:str,):
    """Вывести названия аттрибутов"""
    data = read_file(input_file)
    print(*data.columns)


@parser.add
def all_rules(
    input_file:str, 
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
    """Найти все возможные логические зависимости"""
    data = read_file(input_file)
    rules = get_rules(
        method, data, min_support, min_threshold, width=width, depth=depth,
        n_proc=n_proc, 
    )
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    print_rules(rules, metric, save_to=save_to)


@parser.add
def factor(
    input_file:str, 
    factors:List[str], 
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
    """Найти все логические зависимости для заданных свойств"""
    data = read_file(input_file)
    if isinstance(factors, str):
        factors = factors.split()
        
    data = data[factors]
    rules = get_rules(
        method, data, min_support, min_threshold, n_proc=n_proc,
        width=width, depth=depth,
    )
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    print_rules(rules, metric, save_to=save_to)


@parser.add
def parsel(
    input_file:str, 
    parsel:str,
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
    """Найти все заключения по заданной посылке"""
    data = read_file(input_file)
    rules = get_rules(
        method, data, min_support, min_threshold, n_proc=n_proc,
        width=width, depth=depth,
    )
    rules = tools.filter_rows_postprocessing(rules, parsel, "antecedents")
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    print_rules(rules, metric, save_to=save_to)


@parser.add
def conclusion(
    input_file:str, 
    conclusion:str,
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
    """Найти все посылки по заданноому заключению"""
    data = read_file(input_file)
    rules = get_rules(
        method, data, min_support, min_threshold, n_proc=n_proc,
        width=width, depth=depth,
    )
    rules = tools.filter_rows_postprocessing(rules, conclusion, "consequents")
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    print_rules(rules, metric, save_to=save_to)


@parser.add
def estimate_rule(
    input_file:str,
    parsel:str,
    conclusion:str,
    verbose=False,
    metric="confidence",
    save_to=None,
):
    """Оценить заданные посылку и заключение"""
    pattern = r"[\w\d_-]+\s*(={1,2})\s*(?:\d+|'[\w\d_-]+')([\s,]&\s?)?"
    data = read_file(input_file)
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
    print_rules(rule, metric, save_to=save_to)


def main():
    parser.run()


if __name__ == '__main__':
    main()
