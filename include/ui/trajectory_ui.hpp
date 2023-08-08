//
// Created by weihao on 23-8-8.
//

#ifndef SSVIO_TRAJECTORY_UI_HPP
#define SSVIO_TRAJECTORY_UI_HPP

#include <utility>

#include "pangolin/gl/glvbo.h"
#include "pangolin/gl/gl.h"
#include "sophus/se3.hpp"
#include "glog/logging.h"
#include "Eigen/Core"
#include "Eigen/Dense"

namespace ui {

class TrajectoryUI
{
  public:
  explicit TrajectoryUI(Eigen::Vector3f color)
  : color_(std::move(color))
  {
    position_.reserve(max_size_);
    pose_.reserve(max_size_);
  }
  /// Add a trajectory point
  void AddTrajectoryPose(const Sophus::SE3d &pose);
  /// Render the trajectory
  void Render();
  /// clear the pose and trajectory
  void Clear();
  std::vector<Sophus::SE3f> GetTrajecotryPoses() const { return pose_; };
  private:
  size_t max_size_ = 1e6;
  std::vector<Eigen::Vector3f> position_;           /// trajectory
  std::vector<Sophus::SE3f> pose_;                  /// pose
  Eigen::Vector3f color_ = Eigen::Vector3f::Zero(); /// color
};

} // namespace ui

#endif //SSVIO_TRAJECTORY_UI_HPP
