#define _USE_MATH_DEFINES
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "aabb.h"
#include "brute_force_gpu.h"
#include "barnes_hut.h"
#include "barnes_hut_gpu.h"
#include "kernel.h"
#include "py_tree.h"
#include "py_tree_gpu.h"
#include "sampler.h"
#include "tree2d.h"
#include "tree_sampler.h"
#include "tree_sampler_gpu.h"

namespace py = pybind11;

// NOTE: The kernels do not take function pointers because (to my knowledge),
// you can't pass in a device function pointer from a host function
// (particularly when a function of the same name exists on the host and
// device).

template<int N>
__global__ void evalBruteForceGPU_gravity(const LeafData<N>* leaf_data,
                                          int num_elements,
                                          const FLOAT* queries,
                                          int num_queries, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  // Don't early return - just use leftover threads to load into memory without
  // accumulating a result.

  results[query_idx] =
      bruteForceSummationGPU<FLOAT>(leaf_data, num_elements, queries,
                                     num_queries, gravityPotential, query_idx);
}

template<int N>
__global__ void evalBruteForceGPU_wn(const LeafData<N>* leaf_data,
                                     int num_elements, const FLOAT* queries,
                                     int num_queries, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  // Don't early return - just use leftover threads to load into memory without
  // accumulating a result.

  results[query_idx] =
      bruteForceSummationGPU<FLOAT>(leaf_data, num_elements, queries,
                                     num_queries, signedSolidAngle, query_idx);
}

template <int N>
__global__ void evalBruteForceGPU_smoothdist(const LeafData<N>* leaf_data,
                                             int num_elements,
                                             const FLOAT* queries,
                                             int num_queries, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  // Don't early return - just use leftover threads to load into memory without
  // accumulating a result.

  results[query_idx] = bruteForceSummationGPU<FLOAT>(
      leaf_data, num_elements, queries, num_queries, smoothDistExp, query_idx);
}

template <typename TREETYPE>
__global__ void evalBarnesHutGPUVote_gravity(
    const LeafData<TREETYPE::N>* leaf_data, const TREETYPE* root,
    const TREETYPE* node_buffer, const FLOAT* queries, int num_queries,
    FLOAT beta, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (query_idx >= num_queries) {
    return;
  }

  VectorNd<TREETYPE::N> q;
  q(0) = queries[query_idx];
  q(1) = queries[num_queries + query_idx];
  if (TREETYPE::N == 3) {
    q(2) = queries[2 * num_queries + query_idx]; 
  }

  results[query_idx] = evaluateBarnesHutGPUVote<FLOAT>(
      leaf_data, root, node_buffer, q, beta, gravityPotential);
}

template <typename TREETYPE>
__global__ void evalBarnesHutGPUVote_wn(const LeafData<TREETYPE::N>* leaf_data,
                                        const TREETYPE* root,
                                        const TREETYPE* node_buffer,
                                        const FLOAT* queries, int num_queries,
                                        FLOAT beta, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (query_idx >= num_queries) {
    return;
  }

  VectorNd<TREETYPE::N> q;
  q(0) = queries[query_idx];
  q(1) = queries[num_queries + query_idx];
  if (TREETYPE::N == 3) {
    q(2) = queries[2 * num_queries + query_idx]; 
  }

  results[query_idx] = evaluateBarnesHutGPUVote<FLOAT>(
      leaf_data, root, node_buffer, q, beta, signedSolidAngle);
}

template <typename TREETYPE>
__global__ void evalBarnesHutGPUVote_smoothdist(
    const LeafData<TREETYPE::N>* leaf_data, const TREETYPE* root,
    const TREETYPE* node_buffer, const FLOAT* queries, int num_queries,
    FLOAT beta, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (query_idx >= num_queries) {
    return;
  }

  VectorNd<TREETYPE::N> q;
  q(0) = queries[query_idx];
  q(1) = queries[num_queries + query_idx];
  if (TREETYPE::N == 3) {
    q(2) = queries[2 * num_queries + query_idx]; 
  }

  results[query_idx] = evaluateBarnesHutGPUVote<FLOAT>(
      leaf_data, root, node_buffer, q, beta, smoothDistExp);
}

template <typename TREETYPE>
__global__ void multiLevelPrefixControlVariateGPUQuery_gravity(
    const LeafData<TREETYPE::N>* leaf_data, int num_elements,
    const TREETYPE* root, const TREETYPE* node_buffer,
    const LeafData<TREETYPE::N>* contrib_data, const FLOAT* queries,
    int num_queries, int samples_per_subdomain, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (query_idx >= num_queries) {
    return;
  }

  VectorNd<TREETYPE::N> q;
  q(0) = queries[query_idx];
  q(1) = queries[num_queries + query_idx];
  if (TREETYPE::N == 3) {
    q(2) = queries[2 * num_queries + query_idx];
  }

  results[query_idx] = multiLevelPrefixControlVariateSampleGPUQuery<FLOAT>(
      leaf_data, num_elements, root, node_buffer, contrib_data, q,
      gravityPotential, samples_per_subdomain, 0);
}

template <typename TREETYPE>
__global__ void multiLevelPrefixControlVariateGPUQuery_wn(
    const LeafData<TREETYPE::N>* leaf_data, int num_elements,
    const TREETYPE* root, const TREETYPE* node_buffer,
    const LeafData<TREETYPE::N>* contrib_data, const FLOAT* queries,
    int num_queries, int samples_per_subdomain, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (query_idx >= num_queries) {
    return;
  }

  VectorNd<TREETYPE::N> q;
  q(0) = queries[query_idx];
  q(1) = queries[num_queries + query_idx];
  if (TREETYPE::N == 3) {
    q(2) = queries[2 * num_queries + query_idx]; 
  }

  results[query_idx] = multiLevelPrefixControlVariateSampleGPUQuery<FLOAT>(
      leaf_data, num_elements, root, node_buffer, contrib_data, q,
      signedSolidAngle<TREETYPE::N>, samples_per_subdomain, 0);
}

template <typename TREETYPE>
__global__ void multiLevelPrefixControlVariateGPUQuery_smoothdist(
    const LeafData<TREETYPE::N>* leaf_data, int num_elements,
    const TREETYPE* root, const TREETYPE* node_buffer,
    const LeafData<TREETYPE::N>* contrib_data, const FLOAT* queries,
    int num_queries, int samples_per_subdomain, FLOAT* results) {
  int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (query_idx >= num_queries) {
    return;
  }

  VectorNd<TREETYPE::N> q;
  q(0) = queries[query_idx];
  q(1) = queries[num_queries + query_idx];
  if (TREETYPE::N == 3) {
    q(2) = queries[2 * num_queries + query_idx]; 
  }

  results[query_idx] = multiLevelPrefixControlVariateSampleGPUQuery<FLOAT>(
      leaf_data, num_elements, root, node_buffer, contrib_data, q,
      smoothDistExp, samples_per_subdomain, 0);
}

// TODO: pre-copy query points to GPU if it becomes a bottleneck for large examples
template <typename TREETYPE>
std::pair<DynamicVector, float> evalBruteForce_gravity(
    const TREETYPE& tree, const DynamicMatrix& queries) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  evalBruteForceGPU_gravity<<<num_blocks, thread_size,
                              thread_size * sizeof(LeafData<TREETYPE::N>)>>>(
      tree.leaf_data_ptr, tree.num_points, d_queries_ptr, num_queries,
      d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> evalBruteForce_wn(
    const TREETYPE& tree, const DynamicMatrix& queries) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  evalBruteForceGPU_wn<<<num_blocks, thread_size,
                         thread_size * sizeof(LeafData<TREETYPE::N>)>>>(
      tree.leaf_data_ptr, tree.num_points, d_queries_ptr, num_queries,
      d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> evalBruteForce_smoothdist(
    const TREETYPE& tree, const DynamicMatrix& queries) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  evalBruteForceGPU_smoothdist<<<num_blocks, thread_size,
                                 thread_size * sizeof(LeafData<TREETYPE::N>)>>>(
      tree.leaf_data_ptr, tree.num_points, d_queries_ptr, num_queries,
      d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> evalBarnesHut_gravity(
    const TREETYPE& tree, const DynamicMatrix& queries, FLOAT beta) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaDeviceSetLimit(cudaLimitStackSize,
                     10 * 1024);  // larger stack size for recursion

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  evalBarnesHutGPUVote_gravity<<<num_blocks, thread_size>>>(
      tree.leaf_data_ptr, tree.root_ptr, tree.node_buffer_ptr, d_queries_ptr,
      num_queries, beta, d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> evalBarnesHut_wn(const TREETYPE& tree,
                                                 const DynamicMatrix& queries,
                                                 FLOAT beta) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  evalBarnesHutGPUVote_wn<<<num_blocks, thread_size>>>(
      tree.leaf_data_ptr, tree.root_ptr, tree.node_buffer_ptr, d_queries_ptr,
      num_queries, beta, d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> evalBarnesHut_smoothdist(
    const TREETYPE& tree, const DynamicMatrix& queries, FLOAT beta) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  evalBarnesHutGPUVote_smoothdist<<<num_blocks, thread_size>>>(
      tree.leaf_data_ptr, tree.root_ptr, tree.node_buffer_ptr, d_queries_ptr,
      num_queries, beta, d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> multiLevelPrefixControlVariateSampler_gravity(
    const TREETYPE& tree, const DynamicMatrix& queries,
    int samples_per_subdomain) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  multiLevelPrefixControlVariateGPUQuery_gravity<<<num_blocks, thread_size>>>(
      tree.leaf_data_ptr, tree.num_points, tree.root_ptr,
      tree.node_buffer_ptr, tree.contrib_data_ptr, d_queries_ptr, num_queries,
      samples_per_subdomain, d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float> multiLevelPrefixControlVariateSampler_wn(
    const TREETYPE& tree, const DynamicMatrix& queries,
    int samples_per_subdomain) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  multiLevelPrefixControlVariateGPUQuery_wn<<<num_blocks, thread_size>>>(
      tree.leaf_data_ptr, tree.num_points, tree.root_ptr,
      tree.node_buffer_ptr, tree.contrib_data_ptr, d_queries_ptr, num_queries,
      samples_per_subdomain, d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

template <typename TREETYPE>
std::pair<DynamicVector, float>
multiLevelPrefixControlVariateSampler_smoothdist(const TREETYPE& tree,
                                                 const DynamicMatrix& queries,
                                                 int samples_per_subdomain) {
  int num_queries = queries.rows();
  thrust::device_vector<FLOAT> d_queries(queries.data(),
                                         queries.data() + queries.size());
  const FLOAT* d_queries_ptr = thrust::raw_pointer_cast(d_queries.data());

  thrust::device_vector<FLOAT> d_results(num_queries);
  FLOAT* d_results_ptr = thrust::raw_pointer_cast(d_results.data());

  int thread_size = 32;
  int num_blocks = (num_queries + (thread_size - 1)) / thread_size;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  multiLevelPrefixControlVariateGPUQuery_smoothdist<<<num_blocks,
                                                      thread_size>>>(
      tree.leaf_data_ptr, tree.num_points, tree.root_ptr,
      tree.node_buffer_ptr, tree.contrib_data_ptr, d_queries_ptr, num_queries,
      samples_per_subdomain, d_results_ptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy results back
  DynamicVector results(num_queries);
  thrust::copy(d_results.begin(), d_results.end(), results.data());

  return std::make_pair(results, time_ms);
}

using PyQuadtree1 = PyQuadtree<1, 1>;
using PyQuadtree2 = PyQuadtree<1, 2>;
using PyQuadtree3 = PyQuadtree<1, 3>;
using PyOctree1 = PyOctree<1, 1>;
using PyOctree2 = PyOctree<1, 2>;
using PyOctree3 = PyOctree<1, 3>;

using PyQuadtreeGPU1 = PyQuadtreeGPU<1, 1>;
using PyQuadtreeGPU2 = PyQuadtreeGPU<1, 2>;
using PyQuadtreeGPU3 = PyQuadtreeGPU<1, 3>;
using PyOctreeGPU1 = PyOctreeGPU<1, 1>;
using PyOctreeGPU2 = PyOctreeGPU<1, 2>;
using PyOctreeGPU3 = PyOctreeGPU<1, 3>;

PYBIND11_MODULE(sbh_gpu, m) {
  m.doc() =
      "A plugin to expose differentiable Barnes-Hut implementations and "
      "experiments, on the GPU";

  py::class_<PyQuadtreeGPU1>(m, "QuadtreeGPU1")
      .def(py::init<const PyQuadtree1&>());
  py::class_<PyQuadtreeGPU2>(m, "QuadtreeGPU2")
      .def(py::init<const PyQuadtree2&>());
  py::class_<PyQuadtreeGPU3>(m, "QuadtreeGPU3")
      .def(py::init<const PyQuadtree3&>());
  py::class_<PyOctreeGPU1>(m, "OctreeGPU1")
      .def(py::init<const PyOctree1&>());
  py::class_<PyOctreeGPU2>(m, "OctreeGPU2")
      .def(py::init<const PyOctree2&>());
  py::class_<PyOctreeGPU3>(m, "OctreeGPU3")
      .def(py::init<const PyOctree3&>());

  m.def("eval_brute_force_gravity", evalBruteForce_gravity<PyQuadtreeGPU1>);
  m.def("eval_brute_force_gravity", evalBruteForce_gravity<PyQuadtreeGPU2>);
  m.def("eval_brute_force_gravity", evalBruteForce_gravity<PyQuadtreeGPU3>);
  m.def("eval_brute_force_gravity", evalBruteForce_gravity<PyOctreeGPU1>);
  m.def("eval_brute_force_gravity", evalBruteForce_gravity<PyOctreeGPU2>);
  m.def("eval_brute_force_gravity", evalBruteForce_gravity<PyOctreeGPU3>);
  m.def("eval_brute_force_wn", evalBruteForce_wn<PyQuadtreeGPU1>);
  m.def("eval_brute_force_wn", evalBruteForce_wn<PyQuadtreeGPU2>);
  m.def("eval_brute_force_wn", evalBruteForce_wn<PyQuadtreeGPU3>);
  m.def("eval_brute_force_wn", evalBruteForce_wn<PyOctreeGPU1>);
  m.def("eval_brute_force_wn", evalBruteForce_wn<PyOctreeGPU2>);
  m.def("eval_brute_force_wn", evalBruteForce_wn<PyOctreeGPU3>);
  m.def("eval_brute_force_smoothdist",
        evalBruteForce_smoothdist<PyQuadtreeGPU1>);
  m.def("eval_brute_force_smoothdist",
        evalBruteForce_smoothdist<PyQuadtreeGPU2>);
  m.def("eval_brute_force_smoothdist",
        evalBruteForce_smoothdist<PyQuadtreeGPU3>);
  m.def("eval_brute_force_smoothdist", evalBruteForce_smoothdist<PyOctreeGPU1>);
  m.def("eval_brute_force_smoothdist", evalBruteForce_smoothdist<PyOctreeGPU2>);
  m.def("eval_brute_force_smoothdist", evalBruteForce_smoothdist<PyOctreeGPU3>);

  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyQuadtreeGPU1>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyQuadtreeGPU2>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyQuadtreeGPU3>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyOctreeGPU1>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyOctreeGPU2>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyOctreeGPU3>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyQuadtreeGPU1>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyQuadtreeGPU2>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyQuadtreeGPU3>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyOctreeGPU1>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyOctreeGPU2>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyOctreeGPU3>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyQuadtreeGPU1>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyQuadtreeGPU2>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyQuadtreeGPU3>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyOctreeGPU1>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyOctreeGPU2>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyOctreeGPU3>);

  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyQuadtreeGPU1>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyQuadtreeGPU2>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyQuadtreeGPU3>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyOctreeGPU1>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyOctreeGPU2>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyOctreeGPU3>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyQuadtreeGPU1>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyQuadtreeGPU2>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyQuadtreeGPU3>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyOctreeGPU1>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyOctreeGPU2>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyOctreeGPU3>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyQuadtreeGPU1>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyQuadtreeGPU2>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyQuadtreeGPU3>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyOctreeGPU1>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyOctreeGPU2>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyOctreeGPU3>);
}
