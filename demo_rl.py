import time
import pickle
import numpy as np
import torch
from mushr_env import MushrSim


def to_cuda_tensor(x):
    return torch.from_numpy(x).float().cuda()


def to_numpy(x):
    return x.detach().cpu().numpy()


def main():
    nt = 1000

    # simulator
    mushr_sim = MushrSim()
    obs = to_cuda_tensor(mushr_sim.reset())

    # load controller
    with open("checkpoints/model_001000.p", "rb") as f:
        controller, _, _, _, running = pickle.load(f)
    running.get_tensor()
    running.update_cuda(obs)

    # simulation
    for ti in range(nt):
        print(ti)
        obs_norm = running(obs.reshape([1, 1440]), update=False, use_torch=True).float()
        embed = controller.common(obs_norm)
        u = controller.policy(embed)
        u_np = to_numpy(u)
        u_rl = np.concatenate((np.ones_like(u_np), u_np), axis=-1)   # concat with speed=1, as RL only trains for steering
        obs, label, done, info = mushr_sim.step(u_rl)
        obs = to_cuda_tensor(obs)
    
    mushr_sim.visualization("sim.png")


if __name__ == "__main__":
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2-t1))

