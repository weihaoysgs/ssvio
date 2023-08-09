//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_FRAME_HPP
#define SSVIO_FRAME_HPP

#include "sophus/se3.hpp"
#include "Eigen/Core"
#include "mutex"
#include "opencv2/opencv.hpp"
#include "memory"

namespace ssvio {

class Feature;

class Frame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<Frame> Ptr;
  Frame() = default;
  ~Frame() = default;

  Frame(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp);
  void SetPose(const Sophus::SE3d &pose);
  /// the relative pose to the reference KF
  void SetRelativePose(const Sophus::SE3d &relativePose);
  Sophus::SE3d getPose();
  Sophus::SE3d getRelativePose();

 public:
  unsigned long frame_id_;
  double timestamp_;

  cv::Mat left_image_, right_image_;

  std::vector<std::shared_ptr<Feature>> features_left_;
  std::vector<std::shared_ptr<Feature>> features_right_;

 private:
  Sophus::SE3d pose_; /// just for viewer
  /// for tracking 存储的是相对与上一个关键帧的位姿 T_{c_i {i-t}} 这里我们把 i-t时刻当作关键帧, i为当前时刻
  Sophus::SE3d relative_pose_to_kf_;

  std::mutex update_pose_;
  std::mutex update_realteive_pose_;
};

} // namespace ssvio

#endif //SSVIO_FRAME_HPP
