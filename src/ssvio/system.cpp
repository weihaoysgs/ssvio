#include "ssvio/system.hpp"

namespace ssvio {

System::System(const std::string &config_file_path)
  : sys_config_file_path_(config_file_path)
{
  LOG_ASSERT(!sys_config_file_path_.empty());
  ssvio::Setting::getSingleton()->InitParamSetting(sys_config_file_path_);

  GenerateSteroCamera();
  GenerateORBextractor();

  view_ui_ = std::make_shared<ui::PangolinWindow>();

  frontend_ = std::make_shared<FrontEnd>();
  frontend_->SetCamera(left_camera_, right_camera_);
  frontend_->SetOrbExtractor(orb_extractor_);
  frontend_->SetOrbInitExtractor(orb_init_extractor_);
  frontend_->SetViewUI(view_ui_);

  LOG_ASSERT(view_ui_->Init());
}

bool System::RunStep(const cv::Mat &left_img, const cv::Mat &right_img,
                     const double timestamp)
{
  LOG_ASSERT(!left_img.empty() && !right_img.empty() && timestamp >= 0);
  bool track_success = frontend_->GrabSteroImage(left_img, right_img, timestamp);
  return track_success;
}

void System::GenerateSteroCamera()
{
  // load the camera params from config file
  bool camera_need_undistortion = Setting::Get<float>("Camera.NeedUndistortion");

  float fx_left = Setting::Get<float>("Camera1.fx");
  float fy_left = Setting::Get<float>("Camera1.fy");
  float cx_left = Setting::Get<float>("Camera1.cx");
  float cy_left = Setting::Get<float>("Camera1.cy");
  Eigen::Vector3d t_left = Eigen::Vector3d::Zero();

  float fx_right = Setting::Get<float>("Camera2.fx");
  float fy_right = Setting::Get<float>("Camera2.fy");
  float cx_right = Setting::Get<float>("Camera2.cx");
  float cy_right = Setting::Get<float>("Camera2.cy");
  float bf = Setting::Get<float>("Camera.Base.Line");
  float baseline = bf / fx_right;
  Eigen::Vector3d t_right = Eigen::Vector3d(-baseline, 0, 0);

  cv::Mat dist_coef_left = cv::Mat::zeros(4, 1, CV_32F);
  cv::Mat dist_coef_right = cv::Mat::zeros(4, 1, CV_32F);
  if (camera_need_undistortion)
  {
    float k1_left = Setting::Get<float>("Camera1.k1");
    float k2_left = Setting::Get<float>("Camera1.k2");
    float p1_left = Setting::Get<float>("Camera1.p1");
    float p2_left = Setting::Get<float>("Camera1.p2");
    dist_coef_left.at<float>(0) = k1_left;
    dist_coef_left.at<float>(1) = k2_left;
    dist_coef_left.at<float>(2) = p1_left;
    dist_coef_left.at<float>(3) = p2_left;

    float k1_right = Setting::Get<float>("Camera2.k1");
    float k2_right = Setting::Get<float>("Camera2.k2");
    float p1_right = Setting::Get<float>("Camera2.p1");
    float p2_right = Setting::Get<float>("Camera2.p2");
    dist_coef_right.at<float>(0) = k1_right;
    dist_coef_right.at<float>(1) = k2_right;
    dist_coef_right.at<float>(2) = p1_right;
    dist_coef_right.at<float>(3) = p2_right;
  }
  // set the pose of left camera as identity isometry matrix by default
  left_camera_ =
      std::shared_ptr<ssvio::Camera>(new Camera(fx_left,
                                                fy_left,
                                                cx_left,
                                                cy_left,
                                                0,
                                                Sophus::SE3d(Sophus::SO3d(), t_left),
                                                dist_coef_left));

  right_camera_ =
      std::shared_ptr<ssvio::Camera>(new Camera(fx_right,
                                                fy_right,
                                                cx_right,
                                                cy_right,
                                                baseline,
                                                Sophus::SE3d(Sophus::SO3d(), t_right),
                                                dist_coef_right));
}

void System::GenerateORBextractor()
{
  int num_orb_bew_features = Setting::Get<int>("ORBextractor.nNewFeatures");
  float scale_factor = Setting::Get<float>("ORBextractor.scaleFactor");
  int n_levels = Setting::Get<int>("ORBextractor.nLevels");
  int fIniThFAST = Setting::Get<int>("ORBextractor.iniThFAST");
  int fMinThFAST = Setting::Get<int>("ORBextractor.minThFAST");
  orb_extractor_ = ORBextractor::Ptr(new ORBextractor(
      num_orb_bew_features, scale_factor, n_levels, fIniThFAST, fMinThFAST));

  int num_features_init = Setting::Get<int>("ORBextractor.nInitFeatures");

  orb_init_extractor_ = ORBextractor::Ptr(new ORBextractor(
      num_features_init, scale_factor, n_levels, fIniThFAST, fMinThFAST));
}

} // namespace ssvio