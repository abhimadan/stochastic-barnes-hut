#pragma once

#include <Eigen/Core>

#include <cmath>

#include "utils.h"

template<int N>
struct AABB {
  VectorNd<N> min_corner;
  VectorNd<N> max_corner;

  BOTH AABB() {
    for (int i = 0; i < N; i++) {
      min_corner(i) = INFINITY;
      max_corner(i) = -INFINITY;
    }
  }

  BOTH AABB(VectorNd<N> min_corner, VectorNd<N> max_corner)
      : min_corner(min_corner), max_corner(max_corner) {}

  BOTH VectorNd<N> diagonal() const { return max_corner - min_corner; }

  BOTH VectorNd<N> center() const { return (min_corner + max_corner) / 2; }

  BOTH bool contains(VectorNd<N> p) const {
    return ((min_corner.array() - FP(1e-4)) <= p.array() &&
            p.array() <= (max_corner.array() + FP(1e-4)))
        .all();
  }

  BOTH VectorNi<N> quantize(VectorNd<N> p, int dim_split) const {
    return (dim_split * (p - min_corner).array() /
            (max_corner - min_corner).array())
        .template cast<int>()
        .cwiseMin(dim_split - 1)
        .cwiseMax(0)
        .matrix();
  }
};

