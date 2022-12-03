import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from torch.utils.tensorboard import SummaryWriter
import pickle
import argparse
import time
from datetime import datetime
import numpy as np

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self.is_created=False
        self.buffer = []

    def create_log(self, log_path):
        self.log = open(log_path + "/log.txt", "a", 1)
        self.is_created = True
        for message in self.buffer:
            self.log.write(message)

    def write(self, message, only_file=False):
        if not only_file:
            self._terminal.write(message)
        if self.is_created:
            self.log.write(message)
        else:
            self.buffer.append(message)

    def flush(self):
        pass


def add_parser(parser):
    # TODO(yue)
    parser.add_argument("--root_dir", type=str, default="outputs")
    parser.add_argument("--random_seed", type=int, default=1007, help="Seed for reproducing")
    parser.add_argument("--exp_name", type=str, default="rl_dbg")  # TODO(yue)
    parser.add_argument("--gpus", type=str, default=None)  # TODO(yue)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--dx_mins', type=float, nargs="+", default=[-0.03, -0.06, -0.25, -0.5])
    parser.add_argument('--dx_maxs', type=float, nargs="+", default=[0.03, 0.06, 0.25, 0.5])
    parser.add_argument('--dcfg_min', type=float, default=-0.05)
    parser.add_argument('--dcfg_max', type=float, default=0.05)
    parser.add_argument('--num_sim_steps', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--switch_bonus', type=float, default=100)
    parser.add_argument('--invalid_cost', type=float, default=100)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument("--reward_type", type=str, default="soft")  # TODO(yue)
    return parser


def setup_dirs_and_loggers(args):
    # set up time-related exp name
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    exp_str = args.exp_name    
    args.exp_dir = "%s/g%s_%s"%(args.root_dir, time_str, exp_str)
    args.exp_dir_full = args.exp_dir
    args.src_dir = os.path.join(args.exp_dir, "code")
    args.log_dir = os.path.join(args.exp_dir, "logs")
    args.viz_dir = os.path.join(args.exp_dir, "viz")
    args.ckpt_dir = os.path.join(args.exp_dir, "models") 
    args.model_dir = args.ckpt_dir

    # created related folders
    for dd in [args.src_dir, args.log_dir, args.viz_dir, args.ckpt_dir]:
        os.makedirs(dd, exist_ok=True)

    # move source code to the folder
    for key_dir in ["src", "utils"]:
        for root, dirs, files in os.walk(key_dir):
            os.makedirs(os.path.join(args.src_dir, key_dir, "/".join(root.split("/")[1:])), exist_ok=True)
            for f in files:
                if f[-3:].lower() == '.py':
                    shutil.copy(os.path.join(root, f), os.path.join(args.src_dir, key_dir, "/".join(root.split("/")[1:]), f))

    # set up the logger
    logger = Logger()
    logger.create_log(args.exp_dir)
    sys.stdout = logger
    logger.write("python " + " ".join(sys.argv) + "\n", only_file=True)

    # save args to file
    np.savez(os.path.join(args.exp_dir, "args.npz"), args=args)
    return args


def setup_dir(parser, rest_args=None):
    if rest_args is not None:
        args = parser.parse_args(rest_args)
    else:
        args = parser.parse_args()
    setup_dirs_and_loggers(args)
    log_path = args.exp_dir_full
    model_path = args.model_dir
    args.seed = args.random_seed
    args.pretrained_path = args.pretrained_path
    base_dir = log_path
    writer = SummaryWriter(base_dir)
    return args, log_path, model_path, base_dir, writer


def save(agent, save_path, i_iter):
    from Algorithms.pytorch.A2C.a2c import A2C
    from Algorithms.pytorch.DDPG.ddpg import DDPG
    from Algorithms.pytorch.PPO.ppo import PPO
    from Algorithms.pytorch.SAC.sac import SAC
    from Algorithms.pytorch.TD3.td3 import TD3
    from Algorithms.pytorch.TRPO.trpo import TRPO
    from Algorithms.pytorch.VPG.vpg import VPG
    from Algorithms.pytorch.GAIL.gail import GAIL

    if isinstance(agent, A2C):
        data = (agent.ac_net, agent.running_state)
    elif isinstance(agent, (DDPG, PPO, TRPO, VPG)):
        data = (agent.policy_net, agent.value_net, agent.running_state)
    elif isinstance(agent, SAC):
        data = (agent.policy_net, agent.value_net, agent.q_net_1, agent.q_net_2, agent.running_state)
    elif isinstance(agent, TD3):
        data = (agent.policy_net, agent.value_net_1, agent.value_net_2, agent.running_state)
    elif isinstance(agent, GAIL):
        data = (agent.policy, agent.value, agent.discriminator)
        print("data saved by GAIL", save_path, i_iter)
    else:
        raise NotImplementedError
    
    pickle.dump(data, open('{}/model_{:06d}.p'.format(save_path, i_iter), 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algor', type=str, default=None)
    args, rest = parser.parse_known_args()
    ridx = rest.index("--exp_name")+1
    exp_name = rest[ridx]
    if exp_name.endswith(args.algor)==False:
        if exp_name.endswith("_"):
            rest[ridx] = exp_name + args.algor
        else:
            rest[ridx] = exp_name + "_" + args.algor

    if "--random_seed" in rest:
        seed_idx = rest.index("--random_seed") + 1
        random_seed = int(rest[seed_idx])
    else:
        random_seed = 1007

    print("using random seed", random_seed)
    rest[ridx] = exp_name + "_" + args.algor + "_" + str(random_seed)

    ridx = rest.index("--gpus") + 1

    os.environ["CUDA_VISIBLE_DEVICES"] = rest[ridx]

    if args.algor == "a2c":
        from Algorithms.pytorch.A2C.main import main as main_func
    elif args.algor == "ddpg":
        from Algorithms.pytorch.DDPG.main import main as main_func
    elif args.algor == "ppo":
        from Algorithms.pytorch.PPO.main import main as main_func
    elif args.algor == "sac":
        from Algorithms.pytorch.SAC.main import main as main_func
    elif args.algor == "td3":
        from Algorithms.pytorch.TD3.main import main as main_func
    elif args.algor == "trpo":
        from Algorithms.pytorch.TRPO.main import main as main_func
    elif args.algor == "vpg":
        from Algorithms.pytorch.VPG.main import main as main_func
    elif args.algor == "gail":
        from Algorithms.pytorch.GAIL.main import main as main_func
    else:
        raise NotImplementedError

    main_func(rest)



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))