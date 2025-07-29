#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "leaf_data.h"
#include "utils.h"

// TODO: eventually change this to take in a third "extra kernel data"
// parameter, or perhaps just a generic block of memory that each kernel can
// interpret/ignore as needed.
template <typename T, int N>
using KernelFunc = T (*)(const LeafData<N>&, VectorNd<N>);

// A list of supported kernel functions for communication with the plugin.
enum KernelFuncID {
  GRAVITY,
  WINDING_NUMBER,
  SMOOTH_DIST
};

// Kernel functions
template<int N>
BOTH FLOAT gravityPotential(const LeafData<N>& data, VectorNd<N> q) {
  FLOAT dist = (data.position - q).norm();
  return -data.mass / (dist + FP(1e-1));
}

template<int N>
BOTH FLOAT signedSolidAngle(const LeafData<N>& data, VectorNd<N> q);

template<>
BOTH FLOAT signedSolidAngle<2>(const LeafData<2>& data, VectorNd<2> q) {
  FLOAT dist_sq = (data.position - q).squaredNorm();
  FLOAT q_dot_n = data.normal.dot(data.position - q);
  return data.mass * q_dot_n / (2 * (FLOAT)PI * (dist_sq + FP(1e-5)));
}

template<>
BOTH FLOAT signedSolidAngle<3>(const LeafData<3>& data, VectorNd<3> q) {
  FLOAT dist = (data.position - q).norm();
  FLOAT q_dot_n = data.normal.dot(data.position - q);
  return data.mass * q_dot_n /
         (4 * (FLOAT)PI * (dist * dist * dist + FP(1e-5)));
}

template <int N>
BOTH FLOAT smoothDistExp(const LeafData<N>& data, VectorNd<N> q) {
  FLOAT dist = (data.position - q).norm();
  return exp(-data.alpha * dist);
}

template<int N>
BOTH VectorNd<N> gravityGrad(const LeafData<N>& data, VectorNd<N> q) {
  VectorNd<N> dvec = q - data.position;
  FLOAT dist = dvec.norm();
  return -data.mass / std::pow(dist + FP(1e-1), 3) * dvec;
}

// Fixed frequency, but can bundle in leaf data later
template<int N>
BOTH FLOAT helmholtzKernelX(const LeafData<N>& data, VectorNd<N> q) {
  FLOAT freq = 2;
  FLOAT dist = (data.position - q).norm();
  return data.mass * std::cos(freq * 2 * (FLOAT)PI * dist) /
         (dist + FP(1e-1));
}

template<int N>
BOTH FLOAT helmholtzKernelY(const LeafData<N>& data, VectorNd<N> q) {
  FLOAT freq = 1;
  FLOAT dist = (data.position - q).norm();
  return data.mass * std::sin(freq * 2 * (FLOAT)PI * dist) /
         (dist + FP(1e-1));
}
