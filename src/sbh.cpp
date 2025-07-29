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
#include <thread>
#include <vector>

#include "aabb.h"
#include "barnes_hut.h"
#include "kernel.h"
#include "py_tree.h"
#include "sampler.h"
#include "tree2d.h"
#include "tree_sampler.h"

namespace py = pybind11;
using namespace py::literals;

std::random_device dev;
std::mt19937 rng(dev());

int getNumberOfCores() {
  int num_cores = std::thread::hardware_concurrency();
  constexpr int default_num_cores = 8;
  return num_cores == 0 ? default_num_cores : num_cores;
}

template <typename T, typename TREETYPE>
DynamicVector evalBarnesHut(const TREETYPE& tree, const DynamicMatrix& queries,
                            FLOAT beta, KernelFuncID kernel_func_id) {
  int num_queries = queries.rows();
  DynamicVector results(num_queries);
  Eigen::VectorXi counts(num_queries);

  int num_threads = getNumberOfCores();
  KernelFunc<T, TREETYPE::N> kernel_func = nullptr;
  switch (kernel_func_id) {
  case GRAVITY:
    kernel_func = gravityPotential;
    break;
  case WINDING_NUMBER:
    kernel_func = signedSolidAngle;
    break;
  case SMOOTH_DIST:
    kernel_func = smoothDistExp;
    break;
  default:
    std::cout << "Unsupported kernel function!\n";
  }
  auto eval = [&](int thread_idx) {
    for (int i = thread_idx; i < num_queries; i += num_threads) {
      int num_contribs;
      results(i) = evaluateBarnesHut_stackless<T>(
          tree.leaf_data.data(), tree.tree->root(), tree.tree->node_buffer(),
          queries.row(i), beta, kernel_func, num_contribs);
      counts(i) = num_contribs;
    }
  };
  std::vector<std::thread> eval_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    eval_threads.emplace_back(eval, thread_idx);
  }
  for (auto& t : eval_threads) {
    t.join();
  }
  return results;
}

template <typename TREETYPE>
DynamicVector evalBarnesHut_gravity(const TREETYPE& tree,
                                    const DynamicMatrix& queries, FLOAT beta) {
  return evalBarnesHut<FLOAT>(tree, queries, beta, GRAVITY);
}

template <typename TREETYPE>
DynamicVector evalBarnesHut_wn(const TREETYPE& tree,
                               const DynamicMatrix& queries, FLOAT beta) {
  return evalBarnesHut<FLOAT>(tree, queries, beta, WINDING_NUMBER);
}

template <typename TREETYPE>
DynamicVector evalBarnesHut_smoothdist(const TREETYPE& tree,
                                       const DynamicMatrix& queries,
                                       FLOAT beta) {
  return evalBarnesHut<FLOAT>(tree, queries, beta, SMOOTH_DIST);
}

template <typename T, typename TREETYPE>
std::pair<DynamicVector, Eigen::VectorXi>
multiLevelPrefixControlVariateSampler(const TREETYPE& tree,
                                      const DynamicMatrix& queries,
                                      KernelFuncID kernel_func_id,
                                      std::mt19937& rng,
                                      int samples_per_subdomain) {
  int num_queries = queries.rows();
  DynamicVector results(num_queries);
  Eigen::VectorXi counts(num_queries);

  int num_threads = getNumberOfCores();
  KernelFunc<T, TREETYPE::N> kernel_func = nullptr;
  switch (kernel_func_id) {
  case GRAVITY:
    kernel_func = gravityPotential;
    break;
  case WINDING_NUMBER:
    kernel_func = signedSolidAngle;
    break;
  case SMOOTH_DIST:
    kernel_func = smoothDistExp;
    break;
  default:
    std::cout << "Unsupported kernel function!\n";
  }
  auto eval = [&](int thread_idx) {
    for (int i = thread_idx; i < num_queries; i += num_threads) {
      auto result = multiLevelPrefixControlVariateSample_stackless<T>(
          tree.leaf_data.data(), tree.leaf_data.size(), tree.tree->root(),
          tree.tree->node_buffer(), queries.row(i), kernel_func, rng,
          samples_per_subdomain);
      results(i) = result.first;
      counts(i) = result.second;
    }
  };
  std::vector<std::thread> eval_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    eval_threads.emplace_back(eval, thread_idx);
  }
  for (auto& t : eval_threads) {
    t.join();
  }
  return std::make_pair(results, counts);
}

template <typename TREETYPE>
std::pair<DynamicVector, Eigen::VectorXi>
multiLevelPrefixControlVariateSampler_gravity(const TREETYPE& tree,
                                              const DynamicMatrix& queries,
                                              int samples_per_subdomain) {
  return multiLevelPrefixControlVariateSampler<FLOAT>(
      tree, queries, GRAVITY, rng, samples_per_subdomain);
}

template <typename TREETYPE>
std::pair<DynamicVector, Eigen::VectorXi>
multiLevelPrefixControlVariateSampler_wn(const TREETYPE& tree,
                                         const DynamicMatrix& queries,
                                         int samples_per_subdomain) {
  return multiLevelPrefixControlVariateSampler<FLOAT>(
      tree, queries, WINDING_NUMBER, rng, samples_per_subdomain);
}

template <typename TREETYPE>
std::pair<DynamicVector, Eigen::VectorXi>
multiLevelPrefixControlVariateSampler_smoothdist(const TREETYPE& tree,
                                                 const DynamicMatrix& queries,
                                                 int samples_per_subdomain) {
  return multiLevelPrefixControlVariateSampler<FLOAT>(
      tree, queries, SMOOTH_DIST, rng, samples_per_subdomain);
}

using PyQuadtree1 = PyQuadtree<1, 1>;
using PyQuadtree2 = PyQuadtree<1, 2>;
using PyQuadtree3 = PyQuadtree<1, 3>;

using PyOctree1 = PyOctree<1, 1>;
using PyOctree2 = PyOctree<1, 2>;
using PyOctree3 = PyOctree<1, 3>;

PYBIND11_MODULE(sbh, m) {
  m.doc() =
      "A plugin to expose deterministic and stochastic Barnes-Hut algorithms";

  py::class_<PyQuadtree1>(m, "Quadtree1")
      .def(py::init<const DynamicMatrix&, const DynamicMatrix&,
                    const DynamicVector&, VectorNd<2>, VectorNd<2>, FLOAT>(),
           "points"_a, "normals"_a, "masses"_a, "min_corner"_a, "max_corner"_a,
           "alpha"_a = 1.0);
  py::class_<PyQuadtree2>(m, "Quadtree2")
      .def(py::init<const DynamicMatrix&, const DynamicMatrix&,
                    const DynamicVector&, VectorNd<2>, VectorNd<2>, FLOAT>(),
           "points"_a, "normals"_a, "masses"_a, "min_corner"_a, "max_corner"_a,
           "alpha"_a = 1.0);
  py::class_<PyQuadtree3>(m, "Quadtree3")
      .def(py::init<const DynamicMatrix&, const DynamicMatrix&,
                    const DynamicVector&, VectorNd<2>, VectorNd<2>, FLOAT>(),
           "points"_a, "normals"_a, "masses"_a, "min_corner"_a, "max_corner"_a,
           "alpha"_a = 1.0);
  py::class_<PyOctree1>(m, "Octree1")
      .def(py::init<const DynamicMatrix&, const DynamicMatrix&,
                    const DynamicVector&, VectorNd<3>, VectorNd<3>, FLOAT>(),
           "points"_a, "normals"_a, "masses"_a, "min_corner"_a, "max_corner"_a,
           "alpha"_a = 1.0);
  py::class_<PyOctree2>(m, "Octree2")
      .def(py::init<const DynamicMatrix&, const DynamicMatrix&,
                    const DynamicVector&, VectorNd<3>, VectorNd<3>, FLOAT>(),
           "points"_a, "normals"_a, "masses"_a, "min_corner"_a, "max_corner"_a,
           "alpha"_a = 1.0);
  py::class_<PyOctree3>(m, "Octree3")
      .def(py::init<const DynamicMatrix&, const DynamicMatrix&,
                    const DynamicVector&, VectorNd<3>, VectorNd<3>, FLOAT>(),
           "points"_a, "normals"_a, "masses"_a, "min_corner"_a, "max_corner"_a,
           "alpha"_a = 1.0);

  m.attr("fp") = 8*sizeof(FLOAT);

  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyQuadtree1>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyQuadtree2>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyQuadtree3>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyOctree1>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyOctree2>);
  m.def("eval_barnes_hut_gravity", evalBarnesHut_gravity<PyOctree3>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyQuadtree1>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyQuadtree2>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyQuadtree3>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyOctree1>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyOctree2>);
  m.def("eval_barnes_hut_wn", evalBarnesHut_wn<PyOctree3>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyQuadtree1>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyQuadtree2>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyQuadtree3>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyOctree1>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyOctree2>);
  m.def("eval_barnes_hut_smoothdist", evalBarnesHut_smoothdist<PyOctree3>);

  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyQuadtree1>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyQuadtree2>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyQuadtree3>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyOctree1>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyOctree2>);
  m.def("multi_level_prefix_control_variate_gravity",
        multiLevelPrefixControlVariateSampler_gravity<PyOctree3>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyQuadtree1>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyQuadtree2>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyQuadtree3>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyOctree1>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyOctree2>);
  m.def("multi_level_prefix_control_variate_wn",
        multiLevelPrefixControlVariateSampler_wn<PyOctree3>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyQuadtree1>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyQuadtree2>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyQuadtree3>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyOctree1>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyOctree2>);
  m.def("multi_level_prefix_control_variate_smoothdist",
        multiLevelPrefixControlVariateSampler_smoothdist<PyOctree3>);
}
