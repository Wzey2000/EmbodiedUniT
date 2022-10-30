#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import random

import numpy as np
import torch

# import custom_habitat_baselines
from custom_habitat.config import Config
from custom_habitat_baselines.il.env_based.il_trainer import ILEnvTrainer
from configs.default import get_config
import env_utils # explicitly import to make registry work

os.environ['HABITAT_SIM_LOG'] = "quiet"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default="train",
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--version", default='', type=str)

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--ckpt", default='', type=str)
    parser.add_argument(
        "--debug", default=0, type=int)

    args = parser.parse_args()
    
    if args.version == '':
        version_name = args.cfg.split('/')[-1][:-len(".yaml")] + "_IL"
    else:
        version_name = args.version

    config = get_config(args.cfg, version_name)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    trainer = eval(config.IL_TRAINER_NAME)(config)

    if args.run_type == "train":
        trainer.train(0, args.debug)
    elif args.run_type == "eval":
        trainer.eval(args.ckpt)




if __name__ == "__main__":
    main()
