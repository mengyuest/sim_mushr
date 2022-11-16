# LiDAR MUSHR environment

## Compile the c++ LiDAR library
``` bash
cd range_libc/pywrapper
WITH_CUDA=ON python setup.py build_ext --inplace
```
Test with `python test.py`

## Install the conda packages
1. `conda create -n new_env python==3.7.6 numpy matplotlib opencv-python==4.5.5.64 pillow scipy gym==0.23.1 numba`
2. `conda activate new_env`

## Run the code
1. `python demo.py`