#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Core>

#include "utils.h"

template<typename T>
class ZeroInitializer {
public:
  static BOTH T zero();
};

template<>
BOTH float ZeroInitializer<float>::zero() {
  return 0.0f;
}

template<>
BOTH double ZeroInitializer<double>::zero() {
  return 0.0;
}

template<>
BOTH VectorNd<2> ZeroInitializer<VectorNd<2>>::zero() {
  return VectorNd<2>::Zero();
}

template<>
BOTH VectorNd<3> ZeroInitializer<VectorNd<3>>::zero() {
  return VectorNd<3>::Zero();
}

template<typename T>
class DoublePrecision;

template<>
class DoublePrecision<float> {
public:
  using DoubleType = double;
};

template<>
class DoublePrecision<double> {
public:
  using DoubleType = double;
};

template<>
class DoublePrecision<Eigen::Matrix<float, 2, 1>> {
public:
  using DoubleType = Eigen::Matrix<double, 2, 1>;
};

template<>
class DoublePrecision<Eigen::Matrix<double, 2, 1>> {
public:
  using DoubleType = Eigen::Matrix<double, 2, 1>;
};

template<>
class DoublePrecision<Eigen::Matrix<float, 3, 1>> {
public:
  using DoubleType = Eigen::Matrix<double, 3, 1>;
};

template<>
class DoublePrecision<Eigen::Matrix<double, 3, 1>> {
public:
  using DoubleType = Eigen::Matrix<double, 3, 1>;
};
