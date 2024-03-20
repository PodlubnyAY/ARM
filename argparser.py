import inspect
from argparse import ArgumentParser
from typing import List


class ArgParser:
    def __init__(self, *args, **kwargs):
        self.parser = ArgumentParser()
        self.runner = {}
        self.sub = self.parser.add_subparsers()
        
    def add(self, f):
        info = inspect.getfullargspec(f)
        p = self.sub.add_parser(name=f.__name__, help=f.__doc__)
        if info.defaults:
            defaults = len(info.defaults)
            for arg, val in zip(info.args[::-1], info.defaults[::-1]):
                if isinstance(val, bool):
                    p.add_argument(f"--{arg}",  default=val, action="store_true")
                elif val is None:
                    p.add_argument(f"--{arg}", type=str, default=None,
                                   help="По умолчанию не используется")
                else:
                    p.add_argument(f"--{arg}", type=type(val), default=val,
                                   help=f"по умолчанию {val}")
        else:
            defaults = 0
            
        for i in range(len(info.args) - defaults):
            if info.annotations[info.args[i]] is List[str]:
                p.add_argument(
                    info.args[i], type=str, nargs='+',
                    help="comma separated list",
                )
                continue
            p.add_argument(info.args[i], type=info.annotations[info.args[i]])
        
        self.runner[f.__name__] = f
        p.set_defaults(command=f.__name__, )
        return f
    
    def run(self):
        args = self.parser.parse_args()
        args_d = vars(args)
        func = self.runner[args.command]
        args_d.pop('command', None)
        func(**args_d)
