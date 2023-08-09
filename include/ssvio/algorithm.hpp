//
// Created by weihao on 23-8-9.
//
#ifndef SSVIO_ALGORITHM_HPP
#define SSVIO_ALGORITHM_HPP

#include "sophus/se3.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <algorithm>

namespace ssvio {
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<double, 3, 4> Mat34;
/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<Sophus::SE3d> &poses,
                          const std::vector<Eigen::Vector3d> points,
                          Eigen::Vector3d &pt_world)
{
  MatXX A(2 * poses.size(), 4);
  VecX b(2 * poses.size());
  b.setZero();
  for (size_t i = 0; i < poses.size(); ++i)
  {
    Mat34 m = poses[i].matrix3x4();
    A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
    A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
  }
  auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

  if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2)
  {
    return true;
  }
  /// give up the bad solution
  return false;
}
} // namespace ssvio

#endif //SSVIO_ALGORITHM_HPP