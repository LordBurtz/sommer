"""Train a registered model from the command line (same code path as refac/03_train.ipynb).

  uv run python -m sommer.train --model PrositLike --preliminary          # quick sanity run
  uv run python -m sommer.train --model PeptideEncoder -p hidden_dimensions=128 -p transformer_layers=4 \
      --lr 6e-4 --batch-size 128 --max-epochs 100 --patience 15
  uv run python -m sommer.train --list
Every TrainConfig field is a flag (--max-epochs, --scheduler cosine, ...); -p sets constructor params.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import fields

from sommer.data import load_dataset
from sommer.models import MODELS
from sommer.training import TrainConfig, Trainer


def parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE",
                    help="model constructor argument, repeatable")
    ap.add_argument("--exclude-imputed", action="store_true", help="train on annotated charges only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--note", default=None, help="free text stored in the run record")
    ap.add_argument("--list", action="store_true", help="list registered models and exit")
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            ap.add_argument(flag, action="store_true")
        elif f.name != "scheduler_params":
            ap.add_argument(flag, type=parse_value, default=argparse.SUPPRESS)
    ap.add_argument("--scheduler-param", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    if args.list or not args.model:
        print("registered models:", ", ".join(sorted(MODELS)))
        return None
    config_args = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if hasattr(args, f.name)}
    config_args["scheduler_params"] = dict(kv.split("=", 1) for kv in args.scheduler_param)
    config_args["scheduler_params"] = {k: parse_value(v) for k, v in config_args["scheduler_params"].items()}
    config = TrainConfig(**config_args)
    params = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.param)}

    data = load_dataset(include_imputed=not args.exclude_imputed)
    return Trainer(config, device=args.device).fit(args.model, data, params,
                                                    extra={"note": args.note} if args.note else None).record


if __name__ == "__main__":
    main()
