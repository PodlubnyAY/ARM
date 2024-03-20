import re
import pandas as pd
from typing import List

import tools
import metrics
from argparser import ArgParser

parser = ArgParser()


def read_file(file):
    try:
        df = pd.read_excel(file)
        return df
    except FileNotFoundError:
        print(f"{file} не найден")
        exit(1)


def get_rules(method, df, min_support, min_threshold, tree_root):
    method = tools.METHODS.get(method, tools.fpgrowth)
    base_rules = method(
        df, min_support=min_support,
        min_threshold=min_threshold,
        root=tree_root,
    )
    return base_rules


def print_rules(rules, metric, save_to=None):
    rules = rules.round(2)
    if rules.shape[0]:
        rules = rules.sort_values(by=['support', metric], ascending=False)
        print(rules.to_string(index=False))
        if save_to:
            if save_to.endswith('csv'):
                rules.to_csv(save_to)
            elif save_to.endswith('xlsx'):
                rules.to_excel(save_to)
            else:
                print(f"Неверный формат файла {save_to}. Необходим .csv или .xlsx")
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
    tree_root=None, 
    save_to=None, 
):
    """Найти все возможные логические зависимости"""
    data = read_file(input_file)
    rules = get_rules(method, data, min_support, min_threshold, tree_root)
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
    tree_root=None, 
    save_to=None,
):
    """Найти все логические зависимости для заданных свойств"""
    data = read_file(input_file)
    if isinstance(factors, str):
        factors = factors.split()
        
    data = data[factors]
    rules = get_rules(method, data, min_support, min_threshold, tree_root)
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
    tree_root=None, 
    save_to=None,
):
    """Найти все заключения по заданной посылке"""
    data = read_file(input_file)
    rules = get_rules(method, data, min_support, min_threshold, tree_root)
    rules = tools.filter_rows_postprocessing(data, parsel, "antecedents")
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
    tree_root=None, 
    save_to=None,
):
    """Найти все посылки по заданноому заключению"""
    data = read_file(input_file)
    rules = get_rules(method, data, min_support, min_threshold, tree_root)
    rules = tools.filter_rows_postprocessing(data, conclusion, "consequents")
    if not verbose:
        columns_dropped = list(tools.METRICS - {metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)
    print_rules(rules, metric, save_to=save_to)


@parser.add
def parsel_conclusion(
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
        print(f"Неверно введена посылка: {conclusion}")
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
