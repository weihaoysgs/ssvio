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
#include "ssvio/frontend.hpp"

namespace ssvio {

static std::once_flag singleton_flag;

class Setting
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static std::shared_ptr<Setting> getSingleton()
  {
    std::call_once(singleton_flag, [&] { singleton_ = std::shared_ptr<Setting>(new Setting()); });
    return singleton_;
  }
  bool InitParamSetting(const std::string config_file_path);

  // access the parameter values
  template <typename T> static T Get(const std::string &key)
  {
    return T(Setting::singleton_->file_[key]);
  }
  ~Setting() = default;

 private:
  Setting() = default;

 private:
  static std::shared_ptr<Setting> singleton_;

  cv::FileStorage file_;
};

std::shared_ptr<Setting> Setting::singleton_ = nullptr;

inline bool Setting::InitParamSetting(const std::string config_file_path)
{
  LOG_ASSERT(std::filesystem::exists(config_file_path));
  singleton_->file_ = cv::FileStorage(config_file_path.c_str(),
                                      cv::FileStorage::READ);
  if (!singleton_->file_.isOpened())
  {
    LOG(FATAL) << "parameter file" << config_file_path << "does not exist.";
    singleton_->file_.release();
    return false;
  }
  return true;
}

} // namespace ssvio

#endif //SSVIO_SETTING_HPP
