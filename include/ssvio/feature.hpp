//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_FEATURE_HPP
#define SSVIO_FEATURE_HPP

#include "Eigen/Core"
#include "Eigen/Dense"
#include "memory"
#include "opencv2/opencv.hpp"

class MapPoint;
class KeyFrame;

namespace ssvio {

class Feature
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<Feature> Ptr;
  Feature() = default;
  Feature(const cv::KeyPoint &kp);
  Feature(std::shared_ptr<KeyFrame> kf, const cv::KeyPoint &kp);

 public:
  /// 该特征点所在的关键帧
  std::weak_ptr<KeyFrame> keyframe_;
  cv::KeyPoint kp_position_;
  /// 该特征点提取金字塔之后的点的集合
  std::vector<cv::KeyPoint> pyramid_keypoints_;
  /// 对应的地图3D点
  std::weak_ptr<MapPoint> map_point_;

  bool is_on_left_frame_ = true; // true: on left frame; false: on right frame;
  bool is_outlier_ = false;
};

} // namespace ssvio

#endif //SSVIO_FEATURE_HPP
