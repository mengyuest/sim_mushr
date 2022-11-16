import time
import numpy as np
from mushr_env import MushrSim

def main():
    mushr_sim = MushrSim()
    mushr_sim.reset()

    for ti in range(50):
        v = np.random.uniform(0.5, 1.5, (1, 1))
        w = np.random.uniform(-0.34, 0.34, (1, 1))
        u = np.concatenate((v, w), axis=1)
        mushr_sim.step(u)

    mushr_sim.visualization("sim.png")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds and the screenshot is in sim.png"%(t2 - t1))