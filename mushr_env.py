import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage

import gym
gym.logger.set_level(40)
import range_libc
from numba import jit, njit


def read_map(map_path, bloating=0):
    real_map = cv2.imread(map_path, -1)
    if bloating != 0:
        blur_img_path = map_path.replace(".png", "_blur%d.png" % bloating)
        if os.path.exists(blur_img_path):
            real_map = cv2.imread(blur_img_path, -1)
        else:
            real_map = blur(real_map, bloating=bloating)
            cv2.imwrite(blur_img_path, real_map)

    occ_map = (real_map != 254).astype(dtype=np.float32)
    return real_map, occ_map


def polar_to_xy(radius, angle, shift=[0, 0]):
    x = shift[0] + radius * np.cos(angle)
    y = shift[1] + radius * np.sin(angle)
    return np.stack((x, y), axis=-1)


def transform_center_to_minmax(h, w, cx, cy, scale):
    xmin = cx - w // 2 * scale
    xmax = cx + w // 2 * scale
    ymin = cy - h // 2 * scale
    ymax = cy + h // 2 * scale
    return xmin, xmax, ymin, ymax


def conv(img, kernel):
    h, w = img.shape
    m, n = kernel.shape
    pad_img = np.zeros((h + m - 1, w + n - 1))
    pad_img[m//2: m//2 + h, n//2: n//2 + w] = img
    new_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = np.sum(pad_img[i:i+m, j:j+n] * kernel)
    return new_img


def blur(img, bloating):
    occ_mask = (img==0)
    kernel = np.ones((bloating * 2 + 1, bloating * 2 + 1))
    new_occ_mask = conv(occ_mask, kernel)
    new_img = np.array(img)
    new_img[new_occ_mask>0] = 0
    return new_img


@njit(fastmath=True)
def gen_bbox_ij(s, height, xmin, ymin, scale):
    L = 0.5
    W = 0.3
    ds = np.array([[
            [0.0, 0.0],
            [L/2.0, W/2.0],
            [L/2.0, -W/2.0],
            [-L/2.0, W/2.0],
            [-L/2.0, -W/2.0],
        ]])
    cos = np.cos(s[:, :, 2])
    sin = np.sin(s[:, :, 2])
    new_x = s[:, :, 0] + ds[:, :, 0] * cos - ds[:, :, 1] * sin
    new_y = s[:, :, 1] + ds[:, :, 0] * sin + ds[:, :, 1] * cos

    # return new_x, new_y
    i = height - (new_y - ymin) / scale
    j = (new_x - xmin) / scale
    return i, j


@njit(fastmath=True)
def compute_simple_dynamic(x, y, v, th, dt):
    new_x_sm = x + v * np.cos(th) * dt
    new_y_sm = y + v * np.sin(th) * dt
    new_th_sm = th
    return new_x_sm, new_y_sm, new_th_sm


@njit(fastmath=True)
def compute_complex_dynamic(x, y, v, th, beta, L, dt):
    sin_beta = np.sin(beta)
    new_th_big = th + 2 * v / L * sin_beta * dt
    new_x_big = x + L / (2 * sin_beta) * (np.sin(new_th_big + beta) - np.sin(th + beta))
    new_y_big = y + L / (2 * sin_beta) * (-np.cos(new_th_big + beta) + np.cos(th + beta))
    return new_x_big, new_y_big, new_th_big


@njit(fastmath=True)
def merge_dynamic(mask, new_x_sm, new_y_sm, new_th_sm, new_th_big, new_x_big, new_y_big):
    new_x = mask * new_x_sm + (1 - mask) * new_x_big
    new_y = mask * new_y_sm + (1 - mask) * new_y_big
    new_th = mask * new_th_sm + (1 - mask) * new_th_big
    return new_x, new_y, new_th


@njit(fastmath=True)
def world_to_map(x, y, height, width, xmin, ymin, scale):
    i = height - (y - ymin) / scale
    j = (x - xmin) / scale
    return i, j


@njit(fastmath=True)
def map_to_world_func(ij, height, width, xmin, ymin, scale):
    x = ij[..., 1] * scale + xmin
    y = (height - ij[..., 0]) * scale + ymin
    return x, y


def map_to_world(ij, height, width, xmin, ymin, scale):
    x, y = map_to_world_func(ij, height, width, xmin, ymin, scale)
    return np.stack((x, y), axis=-1)


def randomly_choose_from(IJ_list):
    I_list, J_list = IJ_list
    N = I_list.shape[0]
    idx = np.random.choice(N, 1)[0].item()
    point = np.array([[I_list[idx], J_list[idx]]])
    return point # shape (1, 2)


class DefaultArgs:
    pass


class Lidar:
    def __init__(self, map, xmin, xmax, ymin, ymax, n_readings, scan_method, d_max, img_dmax, theta_disc=None):
        self.n_readings = n_readings
        self.ranges = np.zeros(n_readings, dtype=np.float32)
        self.range_ones = np.ones(n_readings)
        self.theta_range = np.linspace(0, 2 * np.pi, n_readings, endpoint=False)
        self.theta_sym = np.linspace(-np.pi, np.pi, n_readings, endpoint=False)

        self.img_h, img_w = map.shape[:2]
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.img_scale = (self.ymax - self.ymin) / self.img_h
        self.d_max = d_max
        self.img_dmax = img_dmax
        self.buffer = np.zeros(n_readings, dtype=np.float32)

        libc_map = range_libc.PyOMap(map)
        if scan_method == "bl":
            self.algorithm = range_libc.PyBresenhamsLine(libc_map, max_range=d_max)
        elif scan_method == "rm":
            self.algorithm = range_libc.PyRayMarching(libc_map, max_range=d_max)
        elif scan_method == "rmgpu":
            self.algorithm = range_libc.PyRayMarchingGPU(libc_map, max_range=d_max)
        elif scan_method == "cddt":
            self.algorithm = range_libc.PyCDDTCast(libc_map, max_range=d_max, theta_disc=theta_disc)
        elif scan_method == "glt":
            self.algorithm = range_libc.PyGiantLUTCast(libc_map, max_range=d_max, theta_disc=theta_disc)

    def get_bev(self, scan, pcl_ego, render_type="pil", use_point=True):
        dmax = self.img_dmax
        valid = np.where(scan<dmax)[0]
        points = pcl_ego[valid]
        scale = 10
        w = 2 * dmax * scale
        h = 2 * dmax * scale
        xy = np.stack((points[..., 0] * scale + h // 2, h - points[..., 1] * scale - w // 2), axis=-1)

        if render_type == "pil":
            img = Image.new(mode="L", size=(w, h))
            draw = ImageDraw.Draw(img)
            if use_point:
                draw.point(list(xy.flatten()), fill=255)
            else:
                draw.polygon(list(xy.flatten()), fill=255)
            raw_img = np.array(img)
        elif render_type == "cv":
            xy = xy.astype(dtype=np.int32)
            img = np.zeros((h, w))
            if use_point:
                for i in range(xy.shape[0]):
                    cv2.circle(img, (xy[i,0], xy[i,1]), radius=1, color=(255, 255, 255))
            else:
                cv2.fillPoly(img, pts=[xy], color=(255, 255, 255))
            cv2.imwrite("tmp-bev.png", img)
        elif render_type == "libc":
            self.algorithm.saveTrace(b"tmp-bev.png")
        else:
            fig = plt.Figure(figsize=(1,1))
            ax = plt.gca()
            if use_point:
                plt.scatter(points[..., 0], points[..., 1])
            else:
                ax.add_patch(Rectangle((-dmax, -dmax), 2*dmax, 2*dmax, color="gray"))
                ax.add_patch(Polygon(points, facecolor="white", edgecolor="white", alpha=1))
            ax.axis("off")
            plt.axis("scaled")
            ax.set_xlim(-dmax, dmax)
            ax.set_ylim(-dmax, dmax)
            fig.tight_layout()
            plt.savefig("tmp-bev.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
        im = raw_img
        return im

    def get_scan(self, pos, load_bev, noise):
        # real coords -> pixel space
        Is = self.img_h - (pos[1] - self.ymin) / self.img_scale * self.range_ones
        Js = (pos[0] - self.xmin) / self.img_scale * self.range_ones
        theta = pos[2] + np.pi / 2 + self.theta_range
        input_data = np.stack((Is, Js, theta), axis=-1).astype(dtype=np.float32) 
        
        self.algorithm.calc_range_many(input_data, self.buffer)
        scan = self.buffer * self.img_scale
        if noise!=0:
            # addictive noise
            scan = scan + np.uniform(-noise, +noise, scan.shape)
        points = polar_to_xy(scan, theta - np.pi / 2, pos)
        pcl_ego = polar_to_xy(scan, self.theta_sym - np.pi / 2)

        # rasterization
        if load_bev:
            bev = self.get_bev(scan, pcl_ego)
        else:
            bev = None
        return scan, theta, points, bev, pcl_ego  # a series of readings


class MushrSim(gym.Env):
    def __init__(self, args=None, map_path="bravern_floor.png", dt=0.1, noise=0.0, seed=None, bloating=0):
        if args is None:
            args = DefaultArgs()
            args.seed = 1007
            args.viz_last = False
            args.libc_method = "rm"
            args.load_bev = True
            args.n_readings = 720
            args.d_max = 10
            args.theta_disc = 10
            args.n_samples = 1
            args.safe_thresold = 0.05
            args.bloating = 0
            args.free_dist_thres = 1.2
            args.goal_reach_thres = 1.5 
            args.noise = noise
            if dt is not None:
                args.dt = dt
            else:
                args.dt = 0.05
            if seed is not None:
                args.seed = seed
            if bloating != 0:
                args.bloating = bloating
        self.done_checking = "crash"

        np.random.seed(args.seed)
        self.args = args

        self.action_space = gym.spaces.Box(
            low=np.array([-0.5,-0.5]), high=np.array([0.5,0.5]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-10]*720*2), high=np.array([10]*720*2), dtype=np.float32)

        # map configs
        self.real_map, self.occ_map = read_map(map_path, bloating=args.bloating)
        self.height, self.width = self.real_map.shape
        # TODO (HARDCODED!)
        scale = 0.05
        cx = -32.925
        cy = -37.3
        self.xmin, self.xmax, self.ymin, self.ymax = transform_center_to_minmax(self.height, self.width, cx, cy, scale)
        self.scale = (self.ymax - self.ymin) / self.height
        
        # lidar configs
        self.lidar = Lidar(self.occ_map, self.xmin, self.xmax, self.ymin, self.ymax, 
            n_readings=self.args.n_readings, scan_method=self.args.libc_method, 
            d_max=self.args.d_max / self.scale, img_dmax=self.args.d_max, theta_disc = self.args.theta_disc
        )
        
        # free indices configs
        self.dist_field = ndimage.distance_transform_edt(1 - self.occ_map, sampling=0.05)
        self.free_indices = np.where(self.dist_field > args.free_dist_thres)

        # basic info
        self.epi = 0
        self.tidx = 0
        self.history = []
        
    def reset(self, **kwargs):
        self.epi += 1
        self.tidx = 0
        self.history = []
        self.prev_observation = None
        self.prev_label = None

        # TODO make sure the sampled point is always in free space
        self.init_point_map = randomly_choose_from(self.free_indices)
        self.init_point_world = map_to_world(self.init_point_map, self.height, self.width, self.xmin, self.ymin, self.scale)

        th_init = np.random.uniform(-np.pi, np.pi, (1, 1))
        state = np.append(self.init_point_world, th_init, axis=-1)

        # pick destination point
        i = 0
        while i == 0 or np.linalg.norm(self.goal_point_world - self.init_point_world) < 5:
            i+=1
            self.goal_point_map = randomly_choose_from(self.free_indices)
            self.goal_point_world = map_to_world(self.goal_point_map, self.height, self.width, self.xmin, self.ymin, self.scale)

        self.start_state = np.array(state)
        self.state = state
        self.observation = self.get_observation(state)
        self.label = np.min(self.observation["scan"], axis=-1) >= self.args.safe_thresold
        return self.observation["pcl"].flatten()
    
    def action_space_sample(self):
        return np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.low.shape)

    def dynamic(self, s, u):
        x = s[..., 0:1]
        y = s[..., 1:2]
        th = s[..., 2:3]
        v = u[..., 0:1]
        delta = u[..., 1:2]
        # https://github.com/prl-mushr/mushr_base/blob/d20b3d096a13f0e6c4120aed83159d716bbaaca7/mushr_base/src/racecar_state.py#L46
        L = 0.33

        mask = np.abs(delta) < 1e-2
        new_x_sm, new_y_sm, new_th_sm = compute_simple_dynamic(x, y, v, th, self.args.dt)
        beta = np.arctan(0.5 * np.tan(delta))
        beta[beta==0] = 1e-2
        new_x_big, new_y_big, new_th_big = compute_complex_dynamic(x, y, v, th, beta, L, self.args.dt)
        new_x, new_y, new_th = merge_dynamic(mask, new_x_sm, new_y_sm, new_th_sm, new_th_big, new_x_big, new_y_big)
        new_s = np.concatenate((new_x, new_y, new_th), axis=-1)
        return new_s
    
    def get_observation(self, s):
        obs = {"scan": [], "theta": [], "points": [], "bev": [], "pcl":[]}
        for i in range(s.shape[0]):
            scan, theta, points, bev, pcl = self.lidar.get_scan(s[i], load_bev=self.args.load_bev, noise=self.args.noise)
            obs["scan"].append(scan)
            obs["theta"].append(theta)
            obs["points"].append(points)
            obs["bev"].append(bev)
            obs["pcl"].append(pcl)

        for key in obs:
            obs[key] = np.stack(obs[key])
        return obs

    def step(self, action, state=None, **kwargs):
        if state is not None:
            old_state = state
        else:
            old_state = self.state
                 
        new_state = self.dynamic(old_state, action)
        self.prev_observation = self.observation
        self.observation = self.get_observation(new_state)
       
        if "history_less" in kwargs and kwargs["history_less"]==True:
            self.history = []
        else:
            self.history.append(old_state)
        self.state = new_state
        self.prev_label = self.label
        
        label = np.min(self.observation["scan"], axis=-1) >= self.args.safe_thresold

        self.label=label

        reward = 0.1 if label else -3

        info = {"state": old_state, "new_state": new_state}
        if np.all(label==False):
            info["status"] = "crash"
        elif np.all(np.linalg.norm(old_state[:, :2] - self.goal_point_world, axis=-1) <= self.args.goal_reach_thres):
            info["status"] = "reach"
        else:
            info["status"] = "normal"
        if self.done_checking=="crash_or_reach":
            done = info["status"] in ["crash", "reach"]
        else:
            done = info["status"] in ["crash"]
        self.tidx += 1
        return self.observation["pcl"].flatten(), reward, done, info

    def visualization(self, img_dir):
        fs=18
        plt.rcParams.update({'font.size': fs})
        plt.figure(figsize=(16, 16))
        
        plt.imshow(self.real_map, cmap="gray", extent = [self.xmin, self.xmax, self.ymin, self.ymax])
        plt.plot([x[0,0] for x in self.history] + [self.state[0, 0]], 
                [x[0,1] for x in self.history] + [self.state[0,1]], color="red", label="trajs")
        plt.scatter(self.state[0, 0], self.state[0, 1], color="blue", label="ego", s=72)

        points = self.observation["points"][0]

        patch = Polygon(points, color="lightgreen", label="Lidar", alpha=0.9)
        plt.scatter(self.state[0, 0], self.state[0, 1], color="red", label="Mushr", s=96, zorder=1000)
        plt.xlabel("x (m)", fontsize=fs)
        plt.ylabel("y (m)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)       
        ax = plt.gca()
        ax.add_patch(patch)
        plt.legend()
        plt.savefig(img_dir, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def check_collision_map(self, s):  # x in shape (M, 3)
        new_i, new_j = gen_bbox_ij(s[..., None, :], self.height, self.xmin, self.ymin, self.scale)
        new_i, new_j = new_i.astype(dtype=np.int32), new_j.astype(dtype=np.int32)
        new_i = np.clip(new_i, 0, self.height - 1)
        new_j = np.clip(new_j, 0, self.width - 1)
        return np.any(self.occ_map[new_i, new_j] == 1, axis=-1, keepdims=True)