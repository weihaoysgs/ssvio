//
// Created by weihao on 23-8-9.
//

#include "ssvio/frame.hpp"

namespace ssvio {

/// 0~4294967295 0-2^32
static unsigned long FrmaeFactoryId = 0;

Frame::Frame(const cv::Mat &left_image, const cv::Mat &right_image,
             const double &timestamp)
{
  left_image_ = left_image;
  right_image_ = right_image;
  timestamp_ = timestamp;

  frame_id_ = FrmaeFactoryId++;
}

Sophus::SE3d Frame::getPose()
{
  std::unique_lock<std::mutex> lck(update_pose_);
  return pose_;
}

void Frame::SetPose(const Sophus::SE3d &pose)
{
  std::unique_lock<std::mutex> lck(update_pose_);
  pose_ = pose;
}

Sophus::SE3d Frame::getRelativePose()
{
  std::unique_lock<std::mutex> lck(update_realteive_pose_);
  return relative_pose_to_kf_;
}

void Frame::SetRelativePose(const Sophus::SE3d &relative_pose)
{
  std::unique_lock<std::mutex> lck(update_realteive_pose_);
  relative_pose_to_kf_ = relative_pose;
}

} // namespace ssvio