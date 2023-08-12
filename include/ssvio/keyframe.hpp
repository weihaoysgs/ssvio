//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_KEYFRAME_HPP
#define SSVIO_KEYFRAME_HPP

#include "ssvio/frame.hpp"
#include "sophus/se3.hpp"
#include "ssvio/orbvocabulary.hpp"

namespace ssvio {

class MapPoint;

class KeyFrame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<KeyFrame> Ptr;

  KeyFrame() { }

  KeyFrame(std::shared_ptr<Frame> frame);

  static KeyFrame::Ptr CreateKF(std::shared_ptr<Frame> frame);

  void SetPose(const Sophus::SE3d &pose);

  Sophus::SE3d getPose();

  std::vector<cv::KeyPoint> GetKeyPoints();

 public:
  double timestamp_;
  unsigned long frame_id_;
  unsigned long key_frame_id_;

  /// for pose graph optimization
  Sophus::SE3d relative_pose_to_last_KF_;
  Sophus::SE3d relative_pose_to_loop_KF_;
  std::weak_ptr<KeyFrame> loop_key_frame_;
  std::weak_ptr<KeyFrame> last_key_frame_;

  /// pyramid keypoints only for computing ORB descriptors and doing matching
  std::vector<cv::KeyPoint> pyramid_key_points_;
  std::vector<std::shared_ptr<Feature>> features_left_;

  /// DeepLCD::DescrVector mpDescrVector;
  cv::Mat ORBDescriptors_;
  cv::Mat image_left_;

  DBoW2::BowVector bow2_vec_;

 private:
  Sophus::SE3d pose_; /// T_cw

  std::mutex update_get_pose_mutex_;
};

} // namespace ssvio

#endif //SSVIO_KEYFRAME_HPP
