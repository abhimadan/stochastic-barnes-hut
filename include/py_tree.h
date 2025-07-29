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

#include "aabb.h"
#include "tree2d.h"
#include "tree3d.h"
#include "utils.h"

template <uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
struct PYBIND11_EXPORT PyQuadtree {
  static constexpr int N = 2;
  using Tree = Tree2D<LEAF_SIZE, DIM_SPLIT_LOG>;
  using TreeNode = Tree2DNode<LEAF_SIZE, DIM_SPLIT_LOG>;

  std::vector<LeafData<2>> leaf_data;
  std::unique_ptr<Tree> tree;

  PyQuadtree(const DynamicMatrix& points, const DynamicMatrix& normals,
             const DynamicVector& masses, VectorNd<2> min_corner,
             VectorNd<2> max_corner, FLOAT alpha = FP(1.0)) {
    int num_points = points.rows();
    leaf_data.resize(num_points);
    for (int i = 0; i < num_points; i++) {
      leaf_data[i].position = points.row(i);
      leaf_data[i].normal = normals.row(i);
      leaf_data[i].mass = masses(i);
      leaf_data[i].alpha = alpha;
    }
    AABB<2> bounds(min_corner, max_corner);
    // TODO: eventually add this as a parameter for the plugin, but for most
    // practical usage we'll never get close to a tree with this depth
    int max_depth = 100;
    tree = std::make_unique<Tree>(leaf_data, bounds, max_depth);
  }
};

template <uint64_t LEAF_SIZE, uint64_t DIM_SPLIT_LOG>
struct PYBIND11_EXPORT PyOctree {
  static constexpr int N = 3;
  using Tree = Tree3D<LEAF_SIZE, DIM_SPLIT_LOG>;
  using TreeNode = Tree3DNode<LEAF_SIZE, DIM_SPLIT_LOG>;

  std::vector<LeafData<3>> leaf_data;
  std::unique_ptr<Tree> tree;

  PyOctree(const DynamicMatrix& points, const DynamicMatrix& normals,
           const DynamicVector& masses, VectorNd<3> min_corner,
           VectorNd<3> max_corner, FLOAT alpha = FP(1.0)) {
    int num_points = points.rows();
    leaf_data.resize(num_points);
    for (int i = 0; i < num_points; i++) {
      leaf_data[i].position = points.row(i);
      leaf_data[i].normal = normals.row(i);
      leaf_data[i].mass = masses(i);
      leaf_data[i].alpha = alpha;
    }
    AABB<3> bounds(min_corner, max_corner);
    // TODO: eventually add this as a parameter for the plugin, but for most
    // practical usage we'll never get close to a tree with this depth
    int max_depth = 100;
    tree = std::make_unique<Tree>(leaf_data, bounds, max_depth);
  }
};
