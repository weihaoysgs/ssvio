//
// Created by weihao on 23-8-9.
//

#include "ssvio/frontend.hpp"

namespace ssvio {
FrontEnd::FrontEnd()
{
  num_features_init_good_ = Setting::Get<int>("numFeatures.initGood");
  num_features_tracking_good_ = Setting::Get<int>("numFeatures.trackingGood");
  num_features_tracking_bad_ = Setting::Get<int>("numFeatures.trackingBad");
}
void FrontEnd::SetCamera(const Camera::Ptr &left, const Camera::Ptr &right)
{
  assert(left != nullptr && right != nullptr);
  left_camera_ = left;
  right_camera_ = right;
}

void FrontEnd::DetectFeatures() { }

bool FrontEnd::GrabSteroImage(const cv::Mat &left_img, const cv::Mat &right_img,
                              const double timestamp)
{
  current_frame_.reset(new Frame(left_img, right_img, timestamp));

  cv::Mat mask(left_img.size(), CV_8UC1, 255);

  std::vector<cv::KeyPoint> kps;
  cv::Mat show(left_img.size(), left_img.type());
  orb_extractor_->Detect(left_img, mask, kps);

  cv::drawKeypoints(left_img, kps, show);
  cv::imshow("kps", show);
  cv::waitKey(1);
  return true;
}

void FrontEnd::SetOrbExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb)
{
  assert(orb != nullptr);
  orb_extractor_ = orb;
}

void FrontEnd::SetViewUI(const std::shared_ptr<ui::PangolinWindow> &ui)
{
  assert(ui != nullptr);
  view_ui_ = ui;
}

void FrontEnd::SetOrbInitExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb)
{
  assert(orb != nullptr);
  orb_extractor_init_ = orb;
}

} // namespace ssvio
