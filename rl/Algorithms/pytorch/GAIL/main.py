#!/usr/bin/env python
# Created at 2020/3/14
import time

# import click
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Algorithms.pytorch.GAIL.gail import GAIL

# TODO(yue)
import global_patch
import sys
sys.path.append("../../../..")
import argparse

def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
    parser.add_argument("--config_path", type=str, default="src/rl/Algorithms/pytorch/GAIL/config/config.yml",
              help="Model configuration file")
    parser.add_argument("--expert_data_path", type=str, default="/datadrive/mushr_data/mushr_gail_trajs.npz", help="Expert data path")
    parser.add_argument("--render", type=bool, default=False, help="Render environment or not")
    parser.add_argument("--num_process", type=int, default=4, help="Number of process to run environment")
    parser.add_argument("--eval_iter", type=int, default=50, help="Intervals for evaluating model")
    parser.add_argument("--save_iter", type=int, default=50, help="Intervals for saving model")
    parser.add_argument("--max_iter", type=int, default=50, help="Intervals for evaluating model")
    parser.add_argument("--load_model", type=bool, default=False, help="Indicator for whether load trained model")
    parser.add_argument("--load_model_path", type=str, default="trained_models",
              help="Path for loading trained model")
    parser.add_argument("--log_path", type=str, default="./log/", help="Directory to save logs")
    
    # TODO(yue)
    parser = global_patch.add_parser(parser)
    return parser

def main(rest_args=None):
    parser = define_parser()
    args, log_path, model_path, base_dir, writer = global_patch.setup_dir(parser, rest_args)

    base_dir = log_path
    # writer = SummaryWriter(base_dir)

    config = config_loader(path=args.config_path)  # load model configuration

    gail = GAIL(env_id=args.env_id,
                config=config,
                expert_data_path=args.expert_data_path,
                render=args.render,
                num_process=args.num_process,
                seed=args.seed,
                args=args)

    # if load_model:
    #     print(f"Loading Pre-trained GAIL model from {load_model_path}!!!")
    #     gail.load_model(load_model_path)

    for i_iter in range(1, args.max_iter + 1):
        gail.learn(writer, i_iter)

        if i_iter==1 or i_iter % 10 == 0:
            if hasattr(gail.env, "print_stat"):
                gail.env.print_stat()

        if i_iter % args.eval_iter == 0:
            gail.eval(i_iter)

        if i_iter == 1 or i_iter % args.save_iter == 0:
            gail.save_model(model_path)
            global_patch.save(gail, model_path, i_iter)  # TODO(yue)

        torch.cuda.empty_cache()


def config_loader(path=None):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


if __name__ == '__main__':
    main()
