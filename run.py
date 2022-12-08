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
import _init_paths
# import custom_habitat_baselines
from custom_habitat.config import Config
from custom_habitat_baselines.il.env_based.il_trainer import ILEnvTrainer
from configs.default import get_config
import env_utils # explicitly import to make registry work

os.environ['HABITAT_SIM_LOG'] = "quiet"

# python run.py  --run-type eval --cfg ./configs/ObjectNav/CNNRNN/il_objectnav.yaml  --ckpt ./official_object_nav.pth  --split val --debug 1 --video-type disk
# python run.py  --run-type eval --cfg ./configs/ImageNav/CNNRNN/CNNRNN_envbased.yaml  --ckpt ./data/new_checkpoints/CNNRNN_envbased_IL/ckpt99_frame20275712.pth  --split val --video-type disk
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 ./run.py --cfg ./configs/VLN/CNNRNN/VLN_CNNRNN.yaml --debug 1
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
    # For evaluation
    parser.add_argument(
        "--ckpt", default='', type=str)
    parser.add_argument(
        "--split", default='train', type=str) #, choices=['train', 'val', 'test', 'val_seen', 'val_unseen'])
    parser.add_argument(
        "--video-type", default='', choices=['',"disk", "tensorboard"], type=str)
    parser.add_argument(
        "--debug", default=0, type=int)
    parser.add_argument('--gpus',
                        help='gpus id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    # For DDP
    parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
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
        trainer.train(args)
    elif args.run_type == "eval":
        trainer.eval(args)

if __name__ == "__main__":
    main()
