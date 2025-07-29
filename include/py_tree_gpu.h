#pragma once

#include <Eigen/Core>

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include "aabb.h"
#include "py_tree.h"
#include "tree2d.h"
#include "tree3d.h"

template <uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
struct PYBIND11_EXPORT PyQuadtreeGPU {
  static constexpr int N = 2;
  using Tree = Tree2D<LEAF_SIZE, DIM_SPLIT_LOG>;
  using TreeNode = Tree2DNode<LEAF_SIZE, DIM_SPLIT_LOG>;

  thrust::device_vector<LeafData<2>> leaf_data;
  thrust::device_vector<TreeNode> node_buffer;
  thrust::device_vector<LeafData<2>> contrib_data_buffer;

  int num_points;
  const LeafData<2>* leaf_data_ptr;
  const TreeNode* root_ptr;
  const TreeNode* node_buffer_ptr;
  const LeafData<2>* contrib_data_ptr;

  PyQuadtreeGPU(const PyQuadtree<LEAF_SIZE, DIM_SPLIT_LOG>& py_host_tree)
      : leaf_data(py_host_tree.leaf_data.begin(), py_host_tree.leaf_data.end()),
        node_buffer(py_host_tree.tree->node_vector().begin(),
                    py_host_tree.tree->node_vector().end()),
        contrib_data_buffer(py_host_tree.tree->contrib_data_vector().begin(),
                            py_host_tree.tree->contrib_data_vector().end()) {
    num_points = leaf_data.size();
    leaf_data_ptr = thrust::raw_pointer_cast(leaf_data.data());
    root_ptr = thrust::raw_pointer_cast(node_buffer.data());
    node_buffer_ptr = thrust::raw_pointer_cast(node_buffer.data());
    contrib_data_ptr = thrust::raw_pointer_cast(contrib_data_buffer.data());
  }
};

template <uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
struct PYBIND11_EXPORT PyOctreeGPU {
  static constexpr int N = 3;
  using Tree = Tree3D<LEAF_SIZE, DIM_SPLIT_LOG>;
  using TreeNode = Tree3DNode<LEAF_SIZE, DIM_SPLIT_LOG>;

  thrust::device_vector<LeafData<3>> leaf_data;
  thrust::device_vector<TreeNode> node_buffer;
  thrust::device_vector<LeafData<3>> contrib_data_buffer;

  int num_points;
  const LeafData<3>* leaf_data_ptr;
  const TreeNode* root_ptr;
  const TreeNode* node_buffer_ptr;
  const LeafData<3>* contrib_data_ptr;

  PyOctreeGPU(const PyOctree<LEAF_SIZE, DIM_SPLIT_LOG>& py_host_tree)
      : leaf_data(py_host_tree.leaf_data.begin(), py_host_tree.leaf_data.end()),
        node_buffer(py_host_tree.tree->node_vector().begin(),
                    py_host_tree.tree->node_vector().end()),
        contrib_data_buffer(py_host_tree.tree->contrib_data_vector().begin(),
                            py_host_tree.tree->contrib_data_vector().end()) {
    num_points = leaf_data.size();
    leaf_data_ptr = thrust::raw_pointer_cast(leaf_data.data());
    root_ptr = thrust::raw_pointer_cast(node_buffer.data());
    node_buffer_ptr = thrust::raw_pointer_cast(node_buffer.data());
    contrib_data_ptr = thrust::raw_pointer_cast(contrib_data_buffer.data());
  }
};
