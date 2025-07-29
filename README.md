# Stochastic Barnes-Hut

This repo contains a reference implementation for the algorithm described in the
SIGGRAPH 2025 paper, "Stochastic Barnes-Hut Approximation for Fast Summation on
the GPU". The project page can be found
[here](https://www.dgp.toronto.edu/projects/stochastic-barnes-hut/).

The code is packaged as a Python plugin and works both on the CPU and GPU (see
`example.py` to understand how it's used), and with 2D and 3D data. Along with
the stochastic Barnes-Hut algorithm, a deterministic Barnes-Hut implementation
and a fast GPU brute force summation implementation are provided.

## Prerequisites
Make sure you have Python installed, and set up an environment (using, e.g.,
Conda) with the following libraries:
- `numpy`
- `gpytoolbox`
- `libigl`
- `polyscope`
- `robust_laplacian`

## Build Instructions
```
mkdir build
cd build
cmake ..
make -j2
```

This will produce a `.so` library on Unix-based systems (or two on systems with
an NVIDIA GPU), so copy those files to the `python` directory. After that, try
to run `example.py`. Also pass in the `-h` flag to see different options, like
running on the GPU and using different kernels.

On Linux, you may need to run the following command to get your Conda
environment to work correctly (from [this SO
post](https://stackoverflow.com/a/79286982)):
```
conda install -c conda-forge libstdcxx-ng
```

## Future Work
- [ ] Improve plugin architecture to simplify adding new kernel functions from
    C++ 
- [ ] Add support for "global parameters" to kernel functions (e.g., smooth
    distance `alpha` is hacked in by adding it as a tree data member)
- [ ] Unify CPU and GPU plugins into single interface
