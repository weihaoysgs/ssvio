//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_FRONTEND_HPP
#define SSVIO_FRONTEND_HPP

#include "ssvio/camera.hpp"
#include "opencv2/opencv.hpp"
#include "ui/pangolin_window.hpp"
#include "ssvio/setting.hpp"
#include "ssvio/algorithm.hpp"

namespace ssvio {

class Frame;
class ORBextractor;
class Camera;
class Map;
class KeyFrame;
class MapPoint;

enum class FrontendStatus
{
  INITING,
  TRACKING_GOOD,
  TRACKING_BAD,
  LOST
};

class FrontEnd
{
 public:
  FrontEnd();
  ~FrontEnd() = default;
  void SetCamera(const Camera::Ptr &left, const Camera::Ptr &right);
  void SetViewUI(const std::shared_ptr<ui::PangolinWindow> &ui);
  void SetOrbExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb);
  void SetOrbInitExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb);
  void SetMap(const std::shared_ptr<Map> &map);
  int DetectFeatures();
  bool SteroInit();
  int FindFeaturesInRight();
  bool BuidInitMap();
  void InsertKeyFrame();
  bool GrabSteroImage(const cv::Mat &left_img, const cv::Mat &right_img,
                      const double timestamp);

 private:
  FrontendStatus track_status_ = FrontendStatus::INITING;

  std::shared_ptr<Frame> last_frame_ = nullptr;
  std::shared_ptr<KeyFrame> reference_kf_ = nullptr;
  std::shared_ptr<Frame> current_frame_ = nullptr;
  std::shared_ptr<Camera> left_camera_ = nullptr;
  std::shared_ptr<Camera> right_camera_ = nullptr;
  std::shared_ptr<ssvio::ORBextractor> orb_extractor_ = nullptr;
  std::shared_ptr<ssvio::ORBextractor> orb_extractor_init_ = nullptr;
  std::shared_ptr<ui::PangolinWindow> view_ui_ = nullptr;
  std::shared_ptr<Map> map_ = nullptr;

  /// the pose or motion variables of the current frame
  Sophus::SE3d  relative_motion_;  /// T_{c_{i, i-1}}
  Sophus::SE3d relative_motion_to_reference_kf_;

  /// params for deciding the tracking status
  int num_features_tracking_good_;
  int num_features_tracking_bad_;
  int num_features_init_good_;

  bool is_need_undistortion_ = false;
  bool show_orb_detect_result_ = false;
  bool show_lk_result_ = false;
};
} // namespace ssvio

#endif //SSVIO_FRONTEND_HPP
