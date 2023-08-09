//
// Created by weihao on 23-8-9.
//

#include "ssvio/keyframe.hpp"
namespace ssvio {

Sophus::SE3d KeyFrame::getPose()
{
  std::unique_lock<std::mutex> lck(update_get_pose_mutex_);
  return pose_;
}

void KeyFrame::SetPose(const Sophus::SE3d &pose)
{
  std::unique_lock<std::mutex> lck(update_get_pose_mutex_);
  pose_ = pose;
}
} // namespace ssvio
