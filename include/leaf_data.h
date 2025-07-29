#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include "utils.h"

template<int N>
struct LeafData {
  VectorNd<N> position;
  VectorNd<N> normal;
  FLOAT mass;
  FLOAT alpha;

  BOTH LeafData() : mass(0), alpha(0) {}
};
