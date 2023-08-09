//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_SETTING_HPP
#define SSVIO_SETTING_HPP
#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "memory"
#include "mutex"
#include "sophus/se3.hpp"
#include "glog/logging.h"
#include "filesystem"

namespace ssvio {

class Setting
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static std::shared_ptr<Setting> getSingleton();
  bool InitParamSetting(const std::string config_file_path);
  Eigen::Matrix3d getLeftCameraK() const
  {
    Eigen::Matrix3d K;
    K << left_cam_fx_, 0, left_cam_cx_, 0, left_cam_fy_, left_cam_cy_, 0, 0, 1;
    return K;
  }
  ~Setting() = default;

 private:
  Setting() = default;

 private:
  Sophus::SE3d left_camera_pose_;
  Sophus::SE3d right_camera_pose_;
  int cam_width_;
  int cam_height_;
  float left_cam_fx_ = 0.;
  float left_cam_fy_ = 0.;
  float left_cam_cx_ = 0.;
  float left_cam_cy_ = 0.;

  float right_cam_fx_ = 0.;
  float right_cam_fy_ = 0.;
  float right_cam_cx_ = 0.;
  float right_cam_cy_ = 0.;

  float left_cam_k1 = 0.0;
  float left_cam_k2 = 0.0;
  float left_cam_p1 = 0.0;
  float left_cam_p2 = 0.0;

  float right_cam_k1 = 0.0;
  float right_cam_k2 = 0.0;
  float right_cam_p1 = 0.0;
  float right_cam_p2 = 0.0;

  float base_line_ = 0.;
};

static std::shared_ptr<Setting> singleton = nullptr;
static std::once_flag singleton_flag;

std::shared_ptr<Setting> Setting::getSingleton()
{
  std::call_once(singleton_flag, [&] { singleton = std::shared_ptr<Setting>(new Setting()); });
  return singleton;
}

bool Setting::InitParamSetting(const std::string config_file_path)
{
  LOG_ASSERT(std::filesystem::exists(config_file_path));
  cv::FileStorage setting(config_file_path, cv::FileStorage::READ);
  LOG_IF(FATAL, !setting.isOpened()) << config_file_path << "\t Is Open Error";

  setting["Camera1.fx"] >> left_cam_fx_;
  setting["Camera1.fy"] >> left_cam_fy_;
  setting["Camera1.cx"] >> left_cam_cx_;
  setting["Camera1.cy"] >> left_cam_cy_;

  setting["Camera1.k1"] >> left_cam_k1;
  setting["Camera1.k2"] >> left_cam_k2;
  setting["Camera1.p1"] >> left_cam_p1;
  setting["Camera1.p2"] >> left_cam_p2;

  setting["Camera2.fx"] >> right_cam_fx_;
  setting["Camera2.fy"] >> right_cam_fy_;
  setting["Camera2.cx"] >> right_cam_cx_;
  setting["Camera2.cy"] >> right_cam_cy_;

  setting["Camera2.k1"] >> right_cam_k1;
  setting["Camera2.k2"] >> right_cam_k2;
  setting["Camera2.p1"] >> right_cam_p1;
  setting["Camera2.p2"] >> right_cam_p2;

  setting["Camera.width"] >> cam_width_;
  setting["Camera.height"] >> cam_height_;

  setting["Stereo.Base.Line"] >> base_line_;
  setting.release();
  return true;
}

} // namespace ssvio

#endif //SSVIO_SETTING_HPP
