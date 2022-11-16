from mushr_env import MushrSim
import numpy as np

mushr_sim = MushrSim()
mushr_sim.reset()

for ti in range(50):
    v = np.random.uniform(0.5, 1.5, (1, 1))
    w = np.random.uniform(-0.34, 0.34, (1, 1))
    u = np.concatenate((v, w), axis=1)
    mushr_sim.step(u)

mushr_sim.visualization("sim.png")
