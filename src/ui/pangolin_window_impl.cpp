//
// Created by weihao on 23-8-7.
//
#include "ui/pangolin_window_impl.hpp"
#include "ssvio/setting.hpp"

namespace ui {

bool PangolinWindowImpl::InitPangolin()
{
  /// create a window and bind its context to the main thread
  pangolin::CreateWindowAndBind(win_name_, win_width_, win_height_);

  /// 3D mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // unset the current context from the main thread
  pangolin::GetBoundWindow()->RemoveCurrent();

  /// create trajectory
  no_loop_traj_ =
      std::make_unique<ui::TrajectoryUI>(Eigen::Vector3f(0.0, 1.0, 0.0)); /// red color
  loop_traj_ = std::make_unique<ui::TrajectoryUI>(
      Eigen::Vector3f(1.0, 1.0, 51.0 / 255.0)); /// yellow

  //  current_scan_.reset(new PointCloudType);
  //  current_scan_ui_.reset(new ui::UiCloud);
  camera_vo_cloud_ = std::make_unique<ui::CloudUI>(Eigen::Vector3d(1.0, 0, 0),
                                                   ui::CloudUI::UseColor::HEIGHT_COLOR);

  /// data log
  log_yaw_angle_.SetLabels(
      std::vector<std::string>{"yaw_angle", "pitch_angle", "roll_angle"});

  SetDefaultViewImage();
  return true;
}
void PangolinWindowImpl::CreateDisplayLayout()
{
  viewpoint_X_ = ssvio::Setting::Get<float>("Viewer.ViewpointX");
  viewpoint_Y_ = ssvio::Setting::Get<float>("Viewer.ViewpointY");
  viewpoint_Z_ = ssvio::Setting::Get<float>("Viewer.ViewpointZ");
  viewpoint_focus_ = ssvio::Setting::Get<float>("Viewer.Camera.Focus");
  view_axis_direction_ = ssvio::Setting::Get<float>("View.Axis.Direction");
  /// define camera render object (for view / scene browsing)
  auto proj_mat_main = pangolin::ProjectionMatrix(win_width_,
                                                  win_width_,
                                                  viewpoint_focus_,
                                                  viewpoint_focus_,
                                                  win_width_ / 2.,
                                                  win_width_ / 2.,
                                                  cam_z_near_,
                                                  cam_z_far_);

  auto model_view_main = pangolin::ModelViewLookAt(
      viewpoint_X_,
      viewpoint_Y_,
      viewpoint_Z_,
      0,
      0,
      0,
      static_cast<pangolin::AxisDirection>(view_axis_direction_));

  s_cam_main_ =
      pangolin::OpenGlRenderState(std::move(proj_mat_main), std::move(model_view_main));

  /// Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View &d_cam3d_main = pangolin::Display(dis_3d_main_name_)
                                     .SetBounds(0.0, 1.0, 0.0, 1.0)
                                     .SetHandler(new pangolin::Handler3D(s_cam_main_));

  pangolin::View &d_cam3d = pangolin::Display(dis_3d_name_)
                                .SetBounds(0.0, 1.0, 0.0, 0.75)
                                .SetLayout(pangolin::LayoutOverlay)
                                .AddDisplay(d_cam3d_main);

  /// OpenGL 'view' of data. We might have many views of the same data.
  plotter_yam_angle_ =
      std::make_unique<pangolin::Plotter>(&log_yaw_angle_, -10, 100, -M_PI, M_PI, 75, 2);
  plotter_yam_angle_->SetBounds(0., 1 / 3.0f, 0.0f, 0.98f, 752 / 480.);
  plotter_yam_angle_->Track("$i");
  plotter_yam_angle_->SetBackgroundColour(
      pangolin::Colour(248. / 255., 248. / 255., 255. / 255.));

  pangolin::View &cv_img_left = pangolin::Display(left_img_view_name_)
                                    .SetBounds(1 / 3.0f, 2 / 3.0f, 0., 0.98f, 800 / 480.);

  pangolin::View &cv_img_right = pangolin::Display(right_img_view_name_)
                                     .SetBounds(2 / 3.0f, 1.0f, 0.0f, 0.98f, 800 / 480.);

  pangolin::View &d_plot = pangolin::Display(dis_plot_name_)
                               .SetBounds(0.0f, 1.0, 0.70, 1.0)
                               .AddDisplay(*plotter_yam_angle_)
                               .AddDisplay(cv_img_left)
                               .AddDisplay(cv_img_right);

  pangolin::Display(dis_main_name_)
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(menu_width_), 1.0)
      .AddDisplay(d_cam3d)
      .AddDisplay(d_plot);

  /// Create a glTexture container for reading images
  gl_texture_img_left_ = std::make_unique<pangolin::GlTexture>(
      image_width_, image_height_, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
  gl_texture_img_right_ = std::make_unique<pangolin::GlTexture>(
      image_width_, image_height_, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
}

void PangolinWindowImpl::Render()
{
  /// fetch the context and bind it to this thread
  pangolin::BindToContext(win_name_);

  /// Issue specific OpenGl we might need
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  /// menu
  pangolin::CreatePanel("menu").SetBounds(
      0.0, 1.0, 0.0, pangolin::Attach::Pix(menu_width_));
  pangolin::Var<bool> menu_follow_loc("menu.Follow", false, true);
  pangolin::Var<bool> menu_reset_3d_view("menu.Reset 3D View", false, false);
  // pangolin::Var<bool> menu_reset_front_view("menu.Set to front View", false, false);
  pangolin::Var<bool> menu_show_mappoint("menu.Show PointCloud", false, false);
  pangolin::Var<bool> menu_show_trajectory("menu.Show Trajectory", false, false);

  /// display layout
  CreateDisplayLayout();

  while (!pangolin::ShouldQuit() && !exit_flag_)
  {
    /// Clear entire screen
    glClearColor(255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pangolin::Display(dis_3d_main_name_).Activate(s_cam_main_);

    /// menu control
    following_loc_ = menu_follow_loc;

    if (menu_reset_3d_view)
    {
      s_cam_main_.SetModelViewMatrix(pangolin::ModelViewLookAt(
          viewpoint_X_,
          viewpoint_Y_,
          viewpoint_Z_,
          0,
          0,
          0,
          static_cast<pangolin::AxisDirection>(view_axis_direction_)));
      menu_reset_3d_view = false;
    }

    // Render pointcloud

    //    RenderClouds();

    //    DrawCameraFrame(current_pose_, red);
    /// 处理相机跟随问题
    //    if (following_loc_)
    //    {
    //      s_cam_main_.Follow(current_pose_.matrix());
    //    }
    RenderAll();

    /// Swap frames and Process Events
    pangolin::FinishFrame();
  }
  /// unset the current context from the main thread
  pangolin::GetBoundWindow()->RemoveCurrent();
}

/// this fun should be called in other thread
/// @brief the image must be color image
void PangolinWindowImpl::SetViewImage(const cv::Mat &img_left, const cv::Mat &img_right)
{
  assert(!img_left.empty() && img_left.cols > 0 && img_left.rows > 0);
  assert(!img_right.empty() && img_right.cols > 0 && img_right.rows > 0);
  cv::Mat left_dist, right_dist;
  cv::cvtColor(img_left, left_dist, cv::COLOR_GRAY2BGR);
  cv::cvtColor(img_right, right_dist, cv::COLOR_GRAY2BGR);
  std::unique_lock<std::mutex> lck(update_img_mutex_);
  cv::resize(left_dist, left_img_, cv::Size(image_width_, image_height_));
  cv::resize(right_dist, right_img_, cv::Size(image_width_, image_height_));
}

bool PangolinWindowImpl::RenderViewImage()
{
  assert(right_img_.cols > 0 && right_img_.rows > 0 && left_img_.cols > 0 &&
         left_img_.rows > 0);
  {
    std::unique_lock<std::mutex> lck(update_img_mutex_);
    gl_texture_img_right_->Upload(right_img_.data, GL_BGR, GL_UNSIGNED_BYTE);
    gl_texture_img_left_->Upload(left_img_.data, GL_BGR, GL_UNSIGNED_BYTE);
  }
  /// Render images to the window
  pangolin::Display(right_img_view_name_).Activate();
  /// Set the default background color for displaying pictures, it doesn't matter if you don't set it
  glColor3f(1.0f, 1.0f, 1.0f);
  /// The Y axis needs to be reversed, otherwise the output is upside down
  gl_texture_img_right_->RenderToViewportFlipY();

  pangolin::Display(left_img_view_name_).Activate();
  glColor3f(1.0f, 1.0f, 1.0f);
  gl_texture_img_left_->RenderToViewportFlipY();

  return true;
}

void PangolinWindowImpl::SetDefaultViewImage()
{
  std::unique_lock<std::mutex> lck(update_img_mutex_);
  left_img_ = cv::Mat(image_height_, image_width_, CV_8UC3, cv::Scalar(255));
  cv::putText(left_img_,
              "NoImg",
              cv::Point2i(image_width_ / 2 - 60, image_height_ / 2),
              2,
              2,
              cv::Scalar(255, 255, 255));
  right_img_ = cv::Mat(image_height_, image_width_, CV_8UC3, cv::Scalar(128));
  cv::putText(right_img_,
              "NoImg",
              cv::Point2i(image_width_ / 2 - 60, image_height_ / 2),
              2,
              2,
              cv::Scalar(255, 255, 255));
}

void PangolinWindowImpl::RenderAll()
{
  RenderViewImage();
  RenderPlotterDataLog();
  {
    /// when render image, we render a different view which not render point
    pangolin::Display(dis_3d_main_name_).Activate(s_cam_main_);
    std::unique_lock<std::mutex> lck1(update_vo_state_);
    no_loop_traj_->Render();
    std::unique_lock<std::mutex> lck2(update_vo_cloud_);
    camera_vo_cloud_->Render();
  }
}

void PangolinWindowImpl::SetEulerAngle(float yaw, float pitch, float roll)
{
  std::unique_lock<std::mutex> lck(update_euler_angle_mutex_);
  std::get<0>(euler_angle_) = yaw;
  std::get<1>(euler_angle_) = pitch;
  std::get<2>(euler_angle_) = roll;
}

bool PangolinWindowImpl::RenderPlotterDataLog()
{
  std::unique_lock<std::mutex> lck(update_euler_angle_mutex_);
  log_yaw_angle_.Log(
      std::get<0>(euler_angle_), std::get<1>(euler_angle_), std::get<2>(euler_angle_));
  return true;
}

void PangolinWindowImpl::UpdateVisualOdometerState(const Sophus::SE3d &pose)
{
  std::unique_lock<std::mutex> lck(update_vo_state_);
  no_loop_traj_->AddTrajectoryPose(pose);
}

void PangolinWindowImpl::UpdateCloudVOPoint(const Eigen::Vector3d &point)
{
  std::unique_lock<std::mutex> lck(update_vo_cloud_);
  camera_vo_cloud_->AddCloudPoint(point);
}

} // namespace ui