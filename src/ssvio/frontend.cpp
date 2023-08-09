//
// Created by weihao on 23-8-9.
//

#include "ssvio/frontend.hpp"

namespace ssvio {

void FrontEnd::SetCamera(const Camera::Ptr &left, const Camera::Ptr &right)
{
  assert(left != nullptr && right != nullptr);
  left_camera_ = left;
  right_camera_ = right;
}

void FrontEnd::DetectFeatures()
{

}

bool FrontEnd::GrabSteroImage(const cv::Mat &left_img, const cv::Mat &right_img,
                              const double timestamp)
{

  cv::imshow("left", left_img);
  cv::Mat mask(left_img.size(), CV_8UC1, 255);

  std::vector<cv::KeyPoint> kps;
  cv::Mat show;
  orb_extractor_->Detect(left_img, mask, kps);

  cv::drawKeypoints(left_img, kps, show);
  cv::imshow("show_kps", show);
  cv::waitKey(1);
  return true;
}

void FrontEnd::SetOrbExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb)
{
  assert(orb != nullptr);
  orb_extractor_ = orb;
}


} // namespace ssvio
