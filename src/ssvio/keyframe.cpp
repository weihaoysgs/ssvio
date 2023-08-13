//
// Created by weihao on 23-8-9.
//

#include "ssvio/keyframe.hpp"
#include "ssvio/feature.hpp"
#include "ssvio/mappoint.hpp"

namespace ssvio {

KeyFrame::KeyFrame(std::shared_ptr<Frame> frame)
{
  /// set id
  static unsigned long FactoryId = 0;
  key_frame_id_ = FactoryId++;

  /// copy some members form Frame
  frame_id_ = frame->frame_id_;
  timestamp_ = frame->timestamp_;
  image_left_ = frame->left_image_;
  features_left_ = frame->features_left_;
  // mvpFeaturesRight = frame->mvpFeaturesRight; // undesired

  for (size_t i = 0, N = frame->features_left_.size(); i < N; i++)
  {
    auto mp = frame->features_left_[i]->map_point_.lock();
    if (mp != nullptr)
    {
      features_left_[i]->map_point_ = mp;
    }
  }
}

KeyFrame::Ptr KeyFrame::CreateKF(std::shared_ptr<Frame> frame)
{
  KeyFrame::Ptr new_keyframe(new KeyFrame(frame));

  /// link Feature->keyframe_ to the current KF
  /// add the feature to Feature->MapPoint->observation
  for (size_t i = 0, N = new_keyframe->features_left_.size(); i < N; i++)
  {
    auto feat = new_keyframe->features_left_[i];
    feat->keyframe_ = new_keyframe;

    auto mp = feat->map_point_.lock();
    if (mp)
    {
      mp->AddObservation(feat);
      new_keyframe->features_left_[i]->map_point_ = mp;
    }
  }

  return new_keyframe;
}

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

std::vector<cv::KeyPoint> KeyFrame::GetKeyPoints()
{
  std::vector<cv::KeyPoint> kps(features_left_.size());
  for (size_t i = 0, N = features_left_.size(); i < N; i++)
  {
    kps[i] = features_left_[i]->kp_position_;
  }
  return kps;
}
} // namespace ssvio
