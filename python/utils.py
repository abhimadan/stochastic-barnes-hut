import time
from contextlib import contextmanager
import numpy as np


# Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
@contextmanager
def catchtime(name, print_time=True):
    t2 = t1 = time.perf_counter()
    yield lambda: t2 - t1
    t2 = time.perf_counter()
    if print_time:
        print(f'======{name} time: {t2-t1:.6f}s======')


def rescale_for_colormap(w, lower=None, upper=None):
    wmin = w.min() if lower is None else lower
    wmax = w.max() if upper is None else upper
    w_clamped = np.clip(w, wmin, wmax)
    w_scaled = (w_clamped - wmin) / (wmax - wmin)
    w_scaled[np.isneginf(w_scaled)] = 0
    w_scaled[np.isposinf(w_scaled)] = 1
    return w_scaled


def texture_coords_from_field(w, lower=None, upper=None):
    tx = rescale_for_colormap(w, lower, upper)
    return np.stack((tx, np.zeros_like(tx)), axis=1)


# expose params later
def isoline_mask(w):
    # params
    num_isolines = 7
    isoline_width = 0.2
    alpha = 0.3

    freq = 1/num_isolines
    scale = num_isolines/(2*isoline_width)

    show_w = np.round(np.clip(np.fmod(w, freq) * scale, 0, 1))
    return (1-alpha)*show_w + alpha


# Credit to Nick Sharp for providing the original code for this function
def generate_grid_mesh(N_SIDE, ax_ind=2, offset=0, scale=1, plane_offsets=(0,0)):
    EPS = 0.001
    m1EPS = 1. - EPS
    side1, side2 = np.meshgrid(
        scale*np.linspace(-m1EPS,m1EPS,N_SIDE) + plane_offsets[0],
        scale*np.linspace(-m1EPS,m1EPS,N_SIDE) + plane_offsets[1],
        indexing="ij"
    )
    coordstack = [None, None, None]
    coordstack[(ax_ind+1)%3] = side1.ravel()
    coordstack[(ax_ind+2)%3] = side2.ravel()
    coordstack[(ax_ind+0)%3] = offset*np.ones(N_SIDE*N_SIDE)
    grid_verts = np.stack(coordstack, axis=1)

    grid_indices = []
    for i in range(N_SIDE-1):
        for j in range(N_SIDE-1):
            tl = i * N_SIDE + j
            tr = tl+1
            bl = (i+1) * N_SIDE + j
            br = bl + 1

            grid_indices.append((tl, bl, br))
            grid_indices.append((tl, br, tr))

    grid_indices = np.array(grid_indices)

    return grid_verts, grid_indices
