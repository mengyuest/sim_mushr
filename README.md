# LiDAR MUSHR environment

## Install the conda packages
1. `conda create -n new_env -c anaconda python==3.7.6 numpy matplotlib pillow scipy numba Cython`
2. `conda activate new_env`
3. `pip install opencv-python==4.5.5.64 gym==0.23.1`

If shows error: `ImportError: libffi.so.6: cannot open shared object file: No such file or directory`

1. `find /usr/lib -name "libffi.so*"`
2. `sudo ln -s /usr/path/to/libffi.so.7 /usr/lib/path/to/libffi.so.6`

## Compile the c++ LiDAR library
``` bash
cd range_libc_ym/pywrapper
WITH_CUDA=ON pip install -e .
```
1. Test with `python test.py`

## (Back to the source code dir) Run the code
1. `python demo.py`

## To use the pretrained RL controller
1. Install torch: `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`
2. Install rl package: `cd rl && pip install -e . && cd -`
3. Run RL demo: `python demo_rl.py`

## To train the RL controller
1. Run: `python rl/global_patch.py --algor sac --exp_name mushr --gpus 0 --env_id mushr --max_iter 100000 --save_iter 100 --random_seed 1007 --num_process 1`
2. The model checkpoints are saved in `./outputs/gxxxx-xxxxxx_mushr_sac_1007/models/`
