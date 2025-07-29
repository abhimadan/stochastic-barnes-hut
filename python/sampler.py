import numpy as np
import gpytoolbox as gpy


def uniform_circle_particles(num_particles, radius=0.4):
    # only 2d for now
    angles = np.linspace(0, 2*np.pi - 2*np.pi/num_particles, num_particles)
    points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return radius*points, points

def uniform_circle_wn_weights(num_particles, radius=0.4):
    return 2*np.pi*radius/num_particles*np.ones(num_particles)

# Use uniform weights with this too, should be fine if sigma is small enough
def uniform_jittered_circle_particles(num_particles, radius=0.4, sigma=0.05):
    # only 2d for now
    angles = np.linspace(0, 2*np.pi - 2*np.pi/num_particles, num_particles)
    points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    radii = np.random.normal(radius, sigma, num_particles)
    return radii*points, points

def boundary_particle_weights(points):
    points_next = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(points_next - points, axis=1)
    lengths_next = np.roll(lengths, -1)
    return (lengths+lengths_next)/2

def uniform_sample_boundary_curve(num_particles, V):
    Vnext = np.roll(V, -1, axis=0)
    edges = Vnext - V
    J = np.array([[0,1],[-1,0]])
    N = edges @ J
    N = N/(np.linalg.norm(N, axis=1, keepdims=True) + 1e-8)
    lengths = np.linalg.norm(edges, axis=1)
    param = np.cumsum(lengths)
    param_padded = np.concatenate((np.zeros(1), param))
    samples = np.linspace(0, param_padded[-1], num_particles+1)[:-1]
    idxs = np.searchsorted(param_padded, samples)
    interp_t = (samples - param_padded[idxs-1])/(param_padded[idxs] - param_padded[idxs-1])
    interp_t = interp_t[:,None]
    points = (1-interp_t)*V[idxs-1,:] + interp_t*Vnext[idxs-1,:]
    normals = N[idxs-1,:]
    return points, normals

def generate_barycentric_samples(num_particles, dim=2):
    max_iters = 50
    samples = np.random.rand(num_particles, dim)
    rejected_samples = np.sum(samples, axis=1) > 1
    for iter in range(max_iters):
        if not np.any(rejected_samples):
            break

        num_rejected = np.count_nonzero(rejected_samples)
        new_samples = np.random.rand(num_rejected, dim)
        new_rejected_samples = np.sum(new_samples, axis=1) > 1

        rejected_samples_copy = rejected_samples.copy()
        rejected_samples[rejected_samples_copy] = new_rejected_samples
        samples[rejected_samples_copy] = new_samples

    return samples[~rejected_samples,:]

def sample_points_on_mesh(num_particles, V, F, NF):
    barycentric_coords = generate_barycentric_samples(num_particles)
    num_particles = barycentric_coords.shape[0]

    num_faces = F.shape[0]
    areas = gpy.doublearea(V, F)
    areas[np.isnan(areas)] = 0
    probs = areas/np.sum(areas)
    selected_faces = np.random.choice(np.arange(num_faces), size=num_particles, p=probs)

    e1 = V[F[selected_faces,1],:] - V[F[selected_faces,0],:]
    e2 = V[F[selected_faces,2],:] - V[F[selected_faces,0],:]
    samples = V[F[selected_faces,0],:] + barycentric_coords[:,0:1]*e1 + barycentric_coords[:,1:2]*e2

    normals = NF[selected_faces,:]

    return samples, normals
