import argparse
import os
import warnings
from importlib import import_module

warnings.filterwarnings("ignore")


def run_exp() -> None:
    exp_file_list = [e for e in os.listdir("src/exp") if args.exp in e]
    assert len(exp_file_list) == 1
    exp_module: str = os.path.splitext(exp_file_list[0])[0]  # [exp000] -> exp000
    module = import_module(f"src.exp.{exp_module}.main")
    print(f"execute main in src/exp/{exp_module}/main.py")
    module.main()  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp", type=str, required=True, help="experiment filename"
    )
    parser.set_defaults(func=run_exp)
    args = parser.parse_args()
    args.func()
