//
// Created by weihao on 23-8-7.
//

#ifndef SSVIO_PANGOLIN_WINDOW_IMPL_H
#define SSVIO_PANGOLIN_WINDOW_IMPL_H

#include "pangolin/pangolin.h"
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "thread"
#include "atomic"
#include "unistd.h"

#include "ui/trajectory_ui.hpp"
#include "ui/cloud_ui.hpp"

namespace ui {

class PangolinWindowImpl
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  PangolinWindowImpl() = default;
  ~PangolinWindowImpl() = default;
  PangolinWindowImpl(const PangolinWindowImpl &) = delete;
  PangolinWindowImpl &operator=(const PangolinWindowImpl &) = delete;
  PangolinWindowImpl(PangolinWindowImpl &&) = delete;
  PangolinWindowImpl &operator=(PangolinWindowImpl &&) = delete;

  /// Init pangolin
  bool InitPangolin();

  bool DeInit();
  void SetDefaultViewImage();
  /// Render All
  void RenderAll();
  void CreateDisplayLayout();
  void SetViewImage(const cv::Mat &img_left, const cv::Mat &img_right = cv::Mat());
  bool RenderViewImage();
  bool RenderPlotterDataLog();
  void SetEulerAngle(float yaw, float pitch, float roll);
  void UpdateVisualOdometerState(const Sophus::SE3d &pose);
  void UpdateCloudVOPoint(const Eigen::Vector3d &point);

  public:
  std::thread render_thread_;
  std::atomic<bool> exit_flag_ = false;
  std::atomic<bool> vo_result_need_update_ = false;
  std::mutex update_img_mutex_;
  std::mutex update_euler_angle_mutex_;
  std::mutex update_vo_state_;
  std::mutex update_vo_cloud_;

  void Render();

  private:
  /// camera
  pangolin::OpenGlRenderState s_cam_main_;
  /// window layout 
  int win_width_ = 1920;
  int win_height_ = 1080;
  static constexpr float cam_focus_ = 5000;
  static constexpr float cam_z_near_ = 1.0;
  static constexpr float cam_z_far_ = 1e10;
  static constexpr int menu_width_ = 200;
  const std::string win_name_ = "ZJU.S2VIO";
  const std::string dis_main_name_ = "main";
  const std::string dis_3d_name_ = "Cam 3D";
  const std::string dis_3d_main_name_ = "Cam 3D Main"; /// main
  const std::string dis_plot_name_ = "Plot";
  const std::string dis_imgs_name = "Images";
  bool following_loc_ = true; /// follow the camera

  pangolin::DataLog log_yaw_angle_;
  std::unique_ptr<pangolin::Plotter> plotter_yam_angle_ = nullptr;

  int image_width_ = 640;
  int image_height_ = 320;
  std::string left_img_view_name_ = "left_img_view";
  std::string right_img_view_name_ = "right_img_view";

  std::unique_ptr<pangolin::GlTexture> gl_texture_img_left_ = nullptr;
  std::unique_ptr<pangolin::GlTexture> gl_texture_img_right_ = nullptr;

  std::unique_ptr<ui::TrajectoryUI> no_loop_traj_ = nullptr;
  std::unique_ptr<ui::TrajectoryUI> loop_traj_ = nullptr;
  std::unique_ptr<ui::CloudUI> camera_vo_cloud_ = nullptr;

  cv::Mat right_img_;
  cv::Mat left_img_;

  /// yaw pitch roll
  std::tuple<float, float, float> euler_angle_{0, M_PI / 3., -M_PI / 3.};
};

} // namespace ui

#endif // SSVIO_PANGOLIN_WINDOW_IMPL_H
