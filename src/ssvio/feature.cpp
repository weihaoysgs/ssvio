//
// Created by weihao on 23-8-9.
//

#include "ssvio/feature.hpp"

namespace ssvio {
Feature::Feature(std::shared_ptr<KeyFrame> kf, const cv::KeyPoint &kp)
{
  keyframe_ = kf;
  kp_position_ = kp;
}

Feature::Feature(const cv::KeyPoint &kp)
{
  kp_position_ = kp;
}

} // namespace ssvio