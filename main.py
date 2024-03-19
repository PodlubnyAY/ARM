import argparse
import pandas as pd
import tools


def set_args(subparser, name):
    subparser.add_argument('input', type=str, help="Файл с данными для поиска зависимостей")
    subparser.add_argument(
        '-m', '--method', type=str, default='fpgrowth', choices=tools.METHODS,
        help="Выбор метода поиска зависимостей (fpgrowth по умолчанию)"
    )
    subparser.add_argument(
        '-r', type=str, default=None, 
        help="Предполагаемый корневой атрибут (только для метода tree)",
    )
    subparser.add_argument('-v', '--verbose', action='store_true', help="Рассчитать все метрики")
    subparser.add_argument(
        '--metric', type=str, default='confidence', choices=tools.METRICS,
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


def main(args):
    try:
        df = pd.read_excel(args.input)
    except FileNotFoundError:
        print(f"{args.input} не найден")
        return

    if args.command == "factors":
        df = tools.filter_columns(df, args.factors)
    
    method = tools.METHODS.get(args.method, tools.fpgrowth)
    base_rules = method(
        df, min_support=args.min_support,
        min_threshold=args.min_threshold,
        root=args.r,
    )
    rules = tools.ROUTER[args.command](base_rules, args)
    if not args.verbose:
        columns_dropped = list(tools.METRICS - {args.metric})
        rules.drop(columns=columns_dropped, errors='ignore', inplace=True)

    rules = rules.round(2)
    if rules.shape[0]:
        rules = rules.sort_values(by=['support', args.metric], ascending=False)
        print(rules.to_string(index=False))
    else:
        print("Нет логических зависимостей с заданными ограничениями")


if __name__ == '__main__':
    args = parser.parse_args(
        'all --method apriori -r Feature2  --min_threshold 0.69 validate_input.xlsx'.split()
    )
    main(args)
