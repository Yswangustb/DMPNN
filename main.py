# !usr/bin/env python
# -*- encoding: utf-8 -*-
"""入口文件.

@File: main.py
@Time: 2022/07/17 20:35:57
@Author: Crayon112
@SoftWare: VSCode
@Description: 入口文件

"""


import os
from chemprop.train import cross_validate
from chemprop.train.run_training import run_training
from chemprop.args import TrainArgs


def single_round(
    root, dataset, target, mod, opt,
    type="classification",
    order="1.0", save_dir="result",
):
    read_dir = f'data/{dataset}'
    args = [
        "--data_path",
        f"{root}/{read_dir}/{target}.csv",
        "--dataset_type",
        f"{type}",
        "--num_folds",
        "3",
        "--save_preds",
        # '--init_lr',
        # "0.1",
        # "--quiet",
        '--epochs',
        "30",
        "--model_name",
        f"{mod}",
        "--optimizer",
        f"{opt}",
        "--frac_order",
        f"{order}",
        "--save_dir",
        f"{root}/{save_dir}/{type}/{dataset}/{target}/{mod}/{opt}/{order}"
    ]

    cross_validate(TrainArgs().parse_args(args), run_training)



if __name__ == '__main__':
    root = '.'

    round_args = []
    from itertools import product
    for dataset, mod, opt in product(
        ["DNMT1", "HAT", "HDAC", "HDM", "HMT"],
        ["GGNN", "GCN", "MPN"],
        ["Adam", "SGD"],
    ):
        read_dir = f'data/{dataset}'
        cur_drct = os.path.join(root, read_dir)
        for filename in os.listdir(cur_drct):
            target = filename.split('.')[0]
            round_args.append((root, dataset, target, mod, opt))

    for args in round_args:
        single_round(*args)
    # single_round(*(round_args[0]))
