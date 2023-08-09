//
// Created by weihao on 23-8-7.
//
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "iostream"
#include "ui/pangolin_window.hpp"
#include "unistd.h"
#include "common/read_kitii_dataset.hpp"

DEFINE_double(angular_velocity, 10.0, "角速度（角度）制");
DEFINE_double(linear_velocity, 5.0, "车辆前进线速度 m/s");
DEFINE_bool(use_quaternion, false, "是否使用四元数计算");

DEFINE_string(kitti_dataset_path,
              "/home/weihao/dataset/kitti/data_odometry_gray/dataset/sequences/00",
              "kitti dataset path");
using namespace std;

int main(int argc, char **argv)
{
  google::InitGoogleLogging("test_ui");
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  double angular_velocity_rad = FLAGS_angular_velocity * M_PI / 180.0;     // 弧度制角速度
  Sophus::SE3d pose;                                                       // TWB表示的位姿
  Eigen::Vector3d omega(0, 0, angular_velocity_rad);                       // 角速度矢量
  Eigen::Vector3d v_body(FLAGS_linear_velocity, 0, FLAGS_linear_velocity); // 本体系速度
  const double dt = 0.05;                                                  // 每次更新的时间

  ui::PangolinWindow ui;
  LOG_ASSERT(ui.Init()) << "Ui init failed";

  std::string str_sequence_path = fLS::FLAGS_kitti_dataset_path;

  // load sequence frames
  std::vector<std::string> image_left_vec_path, image_right_vec_path;
  std::vector<double> vec_timestamp;
  common::LoadKittiImagesTimestamps(str_sequence_path, image_left_vec_path, image_right_vec_path, vec_timestamp);
  const size_t num_images = image_left_vec_path.size();
  LOG(INFO) << "nImages: " << num_images;

  for (int ni = 0; ni < num_images && !ui.ShouldQuit(); ni++)
  {
    LOG_IF(ERROR, ni % 100 == 99) << "Has processed " << ni + 1 << " frames." << std::endl;
    /// load the frames from database, convert to gray images if rgb images are loaded
    cv::Mat img_left = cv::imread(image_left_vec_path[ni], cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(image_right_vec_path[ni], cv::IMREAD_GRAYSCALE);
    double timestamp = vec_timestamp[ni];
    LOG_IF(FATAL, img_left.empty()) << "Failed to load image at: " << image_left_vec_path[ni];
    ui.ViewImage(img_left, img_right);
    static float val = 0.0;
    ui.PlotAngleValue(std::sin(val++), std::sin(val + 6), std::sin(val + 10));

    /// process each frame
    /// TODO

    /// 更新自身位置
    Eigen::Vector3d v_world = pose.so3() * v_body;
    pose.translation() += v_world * dt;
    /// 更新自身旋转
    pose.so3() = pose.so3() * Sophus::SO3d::exp(omega * dt);
    LOG(INFO) << "pose: " << pose.translation().transpose();
    ui.ShowVisualOdomResult(pose);

    LOG(INFO) << "Process time: " << timestamp << " image";
    usleep(1e4);
  }
  return 0;
}
