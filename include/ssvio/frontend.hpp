//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_FRONTEND_HPP
#define SSVIO_FRONTEND_HPP
#include "ssvio/camera.hpp"
#include "opencv2/opencv.hpp"
#include "ssvio/orbextractor.hpp"

namespace ssvio {
class FrontEnd
{
 public:
  FrontEnd() = default;
  ~FrontEnd() = default;
  void SetCamera(const Camera::Ptr &left, const Camera::Ptr &right);
  void SetOrbExtractor(const std::shared_ptr<ssvio::ORBextractor> & orb);
  void DetectFeatures();
  bool GrabSteroImage(const cv::Mat &left_img, const cv::Mat &right_img,
                      const double timestamp);

 private:
  std::shared_ptr<Camera> left_camera_ = nullptr;
  std::shared_ptr<Camera> right_camera_ = nullptr;
  std::shared_ptr<ssvio::ORBextractor> orb_extractor_ = nullptr;
};
} // namespace ssvio

#endif //SSVIO_FRONTEND_HPP
