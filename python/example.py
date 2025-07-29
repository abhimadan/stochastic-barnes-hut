import numpy as np
import polyscope as ps
import argparse

import sbh
import source
import utils


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu-mode', help='Run algorithms on the GPU', action='store_true')
parser.add_argument('-k', '--kernel', help='Kernel used (0=gravity, 1=wn, 2=smoothdist)', type=int, default=0)
args = parser.parse_args()

num_particles = 1 << 20 if args.gpu_mode else 1 << 15
alpha = 50

S = source.MeshSurface('../data/bunny.obj', scale=1.6)

if args.kernel == 0:
    # Gravity
    points, normals, masses = S.sample(num_particles, source.MassMode.UNIFORM)
    min_corner = np.min(points, axis=0) - 0.01
    max_corner = np.max(points, axis=0) + 0.01

    # Octree1 is better for Barnes-Hut, Octree2 better for stochastic Barnes-Hut
    octree_cpu1 = sbh.Octree1(points, normals, masses, min_corner, max_corner)
    octree_cpu2 = sbh.Octree2(points, normals, masses, min_corner, max_corner)
elif args.kernel == 1:
    # WN
    # Use the winding number mass mode to get solid angle "masses"
    points, normals, masses = S.sample(num_particles, source.MassMode.WN)
    min_corner = np.min(points, axis=0) - 0.01
    max_corner = np.max(points, axis=0) + 0.01

    octree_cpu1 = sbh.Octree1(points, normals, masses, min_corner, max_corner)
    octree_cpu2 = sbh.Octree2(points, normals, masses, min_corner, max_corner)
elif args.kernel == 2:
    # Smooth distance
    points, normals, masses = S.sample(num_particles, source.MassMode.UNIFORM)
    masses = masses * num_particles # rescale to be back to 1
    min_corner = np.min(points, axis=0) - 0.01
    max_corner = np.max(points, axis=0) + 0.01

    # Provide alpha for smooth distance
    octree_cpu1 = sbh.Octree1(points, normals, masses, min_corner, max_corner, alpha)
    octree_cpu2 = sbh.Octree2(points, normals, masses, min_corner, max_corner, alpha)
else:
    print(f'Unsupported mode {args.mode}')
    exit(1)

if args.gpu_mode:
    import sbh_gpu
    octree1 = sbh_gpu.OctreeGPU1(octree_cpu1)
    octree2 = sbh_gpu.OctreeGPU2(octree_cpu2)
else:
    octree1 = octree_cpu1
    octree2 = octree_cpu2

grid_dim = 1000 if args.gpu_mode else 100
eval_grid, grid_faces = utils.generate_grid_mesh(grid_dim, ax_ind=2)
num_query_points = grid_dim * grid_dim

# shuffle eval_grid (https://stackoverflow.com/questions/26577517/inverse-of-random-shuffle)
shuffle_idxs = np.arange(num_query_points)
np.random.shuffle(shuffle_idxs)
unshuffle_idxs = np.zeros_like(shuffle_idxs)
unshuffle_idxs[shuffle_idxs] = np.arange(num_query_points)

eval_grid_shuffled = eval_grid[shuffle_idxs,:]

if args.gpu_mode:
    if args.kernel == 0:
        w_gt, t_gt = sbh_gpu.eval_brute_force_gravity(octree1, eval_grid)
    elif args.kernel == 1:
        w_gt, t_gt = sbh_gpu.eval_brute_force_wn(octree1, eval_grid)
    elif args.kernel == 2:
        w_gt, t_gt = sbh_gpu.eval_brute_force_smoothdist(octree1, eval_grid)
        w_gt = -np.log(w_gt)/alpha
else:
    if args.kernel == 0:
        with utils.catchtime('Brute Force') as t:
            w_gt = sbh.eval_barnes_hut_gravity(octree1, eval_grid, np.inf)
    elif args.kernel == 1:
        with utils.catchtime('Brute Force') as t:
            w_gt = sbh.eval_barnes_hut_wn(octree1, eval_grid, np.inf)
    elif args.kernel == 2:
        with utils.catchtime('Brute Force') as t:
            w_gt = sbh.eval_barnes_hut_smoothdist(octree1, eval_grid, np.inf)
        w_gt = -np.log(w_gt)/alpha
    t_gt = t()
print(f'brute force took {t_gt} ms')

if args.gpu_mode:
    if args.kernel == 0:
        # Warm up kernel for timings
        w_bh_shuffled, _ = sbh_gpu.eval_barnes_hut_gravity(octree1, eval_grid_shuffled, 2)
        _, t_bh = sbh_gpu.eval_barnes_hut_gravity(octree1, eval_grid_shuffled, 2)
    elif args.kernel == 1:
        w_bh_shuffled, _ = sbh_gpu.eval_barnes_hut_wn(octree1, eval_grid_shuffled, 2)
        _, t_bh = sbh_gpu.eval_barnes_hut_wn(octree1, eval_grid_shuffled, 2)
    elif args.kernel == 2:
        w_bh_shuffled, _ = sbh_gpu.eval_barnes_hut_smoothdist(octree1, eval_grid_shuffled, 2)
        _, t_bh = sbh_gpu.eval_barnes_hut_smoothdist(octree1, eval_grid_shuffled, 2)
        w_bh_shuffled = -np.log(w_bh_shuffled)/alpha
else:
    if args.kernel == 0:
        with utils.catchtime('Barnes-Hut') as t:
            w_bh_shuffled = sbh.eval_barnes_hut_gravity(octree1, eval_grid_shuffled, 2)
    elif args.kernel == 1:
        with utils.catchtime('Barnes-Hut') as t:
            w_bh_shuffled = sbh.eval_barnes_hut_wn(octree1, eval_grid_shuffled, 2)
    elif args.kernel == 2:
        with utils.catchtime('Barnes-Hut') as t:
            w_bh_shuffled = sbh.eval_barnes_hut_smoothdist(octree1, eval_grid_shuffled, 2)
        w_bh_shuffled = -np.log(w_bh_shuffled)/alpha
    t_bh = t()
print(f'Barnes-Hut took {t_bh} ms')
w_bh = w_bh_shuffled[unshuffle_idxs]
w_bh_err = np.abs(w_bh - w_gt)
w_bh_log_err = np.log(w_bh_err)
print(f'Barnes-Hut error stats: {w_bh_log_err.min()}-{w_bh_log_err.max()}')

if args.gpu_mode:
    if args.kernel == 0:
        w_mlpcv_shuffled, _ = sbh_gpu.multi_level_prefix_control_variate_gravity(octree2, eval_grid_shuffled, 1)
        _, t_mlpcv = sbh_gpu.multi_level_prefix_control_variate_gravity(octree2, eval_grid_shuffled, 1)
    elif args.kernel == 1:
        w_mlpcv_shuffled, _ = sbh_gpu.multi_level_prefix_control_variate_wn(octree2, eval_grid_shuffled, 1)
        _, t_mlpcv = sbh_gpu.multi_level_prefix_control_variate_wn(octree2, eval_grid_shuffled, 1)
    elif args.kernel == 2:
        w_mlpcv_shuffled, _ = sbh_gpu.multi_level_prefix_control_variate_smoothdist(octree2, eval_grid_shuffled, 1)
        _, t_mlpcv = sbh_gpu.multi_level_prefix_control_variate_smoothdist(octree2, eval_grid_shuffled, 1)
        w_mlpcv_shuffled = -np.log(w_mlpcv_shuffled)/alpha
else:
    if args.kernel == 0:
        with utils.catchtime('Stochastic Barnes-Hut') as t:
            w_mlpcv_shuffled, _ = sbh.multi_level_prefix_control_variate_gravity(octree2, eval_grid_shuffled, 1)
    elif args.kernel == 1:
        with utils.catchtime('Stochastic Barnes-Hut') as t:
            w_mlpcv_shuffled, _ = sbh.multi_level_prefix_control_variate_wn(octree2, eval_grid_shuffled, 1)
    elif args.kernel == 2:
        with utils.catchtime('Stochastic Barnes-Hut') as t:
            w_mlpcv_shuffled, _ = sbh.multi_level_prefix_control_variate_smoothdist(octree2, eval_grid_shuffled, 1)
        w_mlpcv_shuffled = -np.log(w_mlpcv_shuffled)/alpha
    t_mlpcv = t()
print(f'Stochastic Barnes-Hut took {t_mlpcv} ms')
w_mlpcv = w_mlpcv_shuffled[unshuffle_idxs]
w_mlpcv_err = np.abs(w_mlpcv - w_gt)
w_mlpcv_log_err = np.log(w_mlpcv_err)
print(f'Stochastic Barnes-Hut error stats: {w_mlpcv_log_err.min()}-{w_mlpcv_log_err.max()}')

log_err_all = np.concatenate((w_bh_log_err[~np.isinf(w_bh_log_err)], w_mlpcv_log_err[~np.isinf(w_mlpcv_log_err)]))
log_err_min = np.min(log_err_all)
log_err_max = np.max(log_err_all)

ps.init()

V = S.V
F = S.F
ps.register_surface_mesh('mesh', V, F)

ps_slice = ps.register_surface_mesh(f"slice_full", eval_grid, grid_faces, transparency=0.9)
ps_slice.add_scalar_quantity('gravitational potential (GT)', w_gt, enabled=True)
ps_slice.add_scalar_quantity('gravitational potential (BH)', w_bh)
ps_slice.add_scalar_quantity('gravitational potential (SBH)', w_mlpcv)
ps_slice.add_scalar_quantity('gravitational potential log err (BH)', w_bh_log_err, vminmax=(log_err_min,log_err_max))
ps_slice.add_scalar_quantity('gravitational potential log err (SBH)', w_mlpcv_log_err, vminmax=(log_err_min,log_err_max))

ps.show()
