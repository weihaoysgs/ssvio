//
// Created by weihao on 23-8-9.
//

#include "ssvio/frontend.hpp"
#include "ssvio/frame.hpp"
#include "ssvio/feature.hpp"
#include "ssvio/orbextractor.hpp"
#include "ssvio/mappoint.hpp"
#include "ssvio/keyframe.hpp"

namespace ssvio {
FrontEnd::FrontEnd()
{
  num_features_init_good_ = Setting::Get<int>("numFeatures.initGood");
  num_features_tracking_good_ = Setting::Get<int>("numFeatures.trackingGood");
  num_features_tracking_bad_ = Setting::Get<int>("numFeatures.trackingBad");
  is_need_undistortion_ = Setting::Get<int>("Camera.NeedUndistortion");
  show_orb_detect_result_ = Setting::Get<int>("View.ORB.Extractor.Result");
  show_lk_result_ = Setting::Get<int>("View.LK.Folw");
}
void FrontEnd::SetCamera(const Camera::Ptr &left, const Camera::Ptr &right)
{
  assert(left != nullptr && right != nullptr);
  left_camera_ = left;
  right_camera_ = right;
}

bool FrontEnd::GrabSteroImage(const cv::Mat &left_img, const cv::Mat &right_img,
                              const double timestamp)
{
  current_frame_.reset(new Frame(left_img, right_img, timestamp));
  /// undistort the images, which is not required in KITTI
  if (is_need_undistortion_)
  {
    left_camera_->UndistortImage(current_frame_->left_image_,
                                 current_frame_->left_image_);
    right_camera_->UndistortImage(current_frame_->right_image_,
                                  current_frame_->right_image_);
  }

  {
    // std::unique_lock<std::mutex> lck(_mpMap->mmutexMapUpdate);
    switch (track_status_)
    {
    case FrontendStatus::INITING:
      {
        SteroInit();
        break;
      }
    case FrontendStatus::TRACKING_BAD:
    case FrontendStatus::TRACKING_GOOD:
      {
        break;
      }
    case FrontendStatus::LOST:
      {
        break;
      }
    }
  }

  last_frame_ = current_frame_;
  return true;
}

int FrontEnd::DetectFeatures()
{
  cv::Mat mask(current_frame_->left_image_.size(), CV_8UC1, cv::Scalar::all(255));
  for (const auto &feat : current_frame_->features_left_)
  {
    cv::rectangle(mask,
                  feat->kp_position_.pt - cv::Point2f(20, 20),
                  feat->kp_position_.pt + cv::Point2f(20, 20),
                  0,
                  cv::FILLED);
  }

  std::vector<cv::KeyPoint> kps;
  if (track_status_ == FrontendStatus::INITING)
  {
    orb_extractor_init_->Detect(current_frame_->left_image_, mask, kps);
  }
  else
  {
    orb_extractor_->Detect(current_frame_->left_image_, mask, kps);
  }
  int cnt_detected = 0;
  for (const auto &kp : kps)
  {
    Feature::Ptr feature(new Feature(kp));
    current_frame_->features_left_.emplace_back(feature);
    cnt_detected++;
  }
  if (show_orb_detect_result_ && cnt_detected > 0)
  {
    cv::Mat show_img(current_frame_->left_image_.size(), CV_8UC1);
    cv::drawKeypoints(current_frame_->left_image_, kps, show_img);
    cv::imshow("orb_detect_result", show_img);
    cv::waitKey(1);
  }
  return cnt_detected;
}

void FrontEnd::FindFeaturesInRight()
{
  std::vector<cv::Point2f> left_cam_kps, right_cam_kps;
  left_cam_kps.reserve(current_frame_->features_left_.size());
  right_cam_kps.reserve(current_frame_->features_left_.size());
  for (const auto &feat : current_frame_->features_left_)
  {
    left_cam_kps.push_back(feat->kp_position_.pt);
    auto pt_in_map = feat->map_point_.lock();
    if (pt_in_map)
    {
      /// 如果是已经被三角化后的点,则通过第一个相机的位姿和相机之间的外参投影到第二个相机的平面上去
      Eigen::Vector2d p = right_camera_->world2pixel(pt_in_map->getPosition(),
                                                     current_frame_->getRelativePose() *
                                                         reference_kf_->getPose());
      right_cam_kps.push_back(cv::Point2f(p.x(), p.y()));
    }
    else
    {
      right_cam_kps.push_back(feat->kp_position_.pt);
    }
  }

  // LK flow: calculate the keypoints' positions in right frame
  std::vector<uchar> status;
  cv::Mat error;
  cv::calcOpticalFlowPyrLK(
      current_frame_->left_image_,
      current_frame_->right_image_,
      left_cam_kps,
      right_cam_kps,
      status,
      error,
      cv::Size(11, 11),
      3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  /// create new feature objects in right frame
  int num_good_points = 0;
  current_frame_->features_right_.reserve(status.size());
  for (size_t i = 0; i < status.size(); ++i)
  {
    if (status[i])
    {
      /// only the position of keypoint is needed, so size 7 is just for creation with no meaning
      cv::KeyPoint kp(right_cam_kps[i], 7);
      Feature::Ptr feat(new Feature(kp));
      current_frame_->features_right_.push_back(feat);
      num_good_points++;
    }
    else
    {
      current_frame_->features_right_.push_back(nullptr);
    }
  }

  if (show_lk_result_)
  {
    cv::Mat show;
    cv::cvtColor(current_frame_->left_image_, show, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < current_frame_->features_left_.size(); i++)
    {
      if (status[i])
      {
        cv::Point2i pt1 = current_frame_->features_left_[i]->kp_position_.pt;
        cv::Point2i pt2 = current_frame_->features_right_[i]->kp_position_.pt;
        cv::circle(show, pt1, 2, cv::Scalar(0, 250, 0), 2);
        cv::line(show, pt1, pt2, cv::Scalar(0, 0, 255),1.5);
      }
    }
    cv::imshow("LK", show);
    cv::waitKey(1);
  }
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

bool FrontEnd::SteroInit()
{
  DetectFeatures();
  FindFeaturesInRight();
  return true;
}

} // namespace ssvio
