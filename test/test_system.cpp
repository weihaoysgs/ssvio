//
// Created by weihao on 23-8-9.
//
#include "ssvio/system.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "common/read_kitii_dataset.hpp"
#include "chrono"

DEFINE_string(config_yaml_path, "/home/weihao/codespace/ssvio/config/kitti_00.yaml",
              "System config file path");
DEFINE_string(kitti_dataset_path,
              "/home/weihao/dataset/kitti/data_odometry_gray/dataset/sequences/00",
              "kitti dataset path");

int main(int argc, char **argv)
{
  google::InitGoogleLogging("test_system");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags(&argc, &argv, true);

  /// load sequence frames
  std::vector<std::string> image_left_vec_path, image_right_vec_path;
  std::vector<double> vec_timestamp;
  common::LoadKittiImagesTimestamps(fLS::FLAGS_kitti_dataset_path,
      image_left_vec_path, image_right_vec_path, vec_timestamp);
  const size_t num_images = image_left_vec_path.size();
  LOG(INFO) << "Num Images: " << num_images;

  /// Init SLAM System
  ssvio::System system(fLS::FLAGS_config_yaml_path);

  for (int ni = 0; ni < num_images && !system.getViewUi()->ShouldQuit(); ni++)
  {
    LOG_IF(ERROR, ni % 100 == 99) << "Has processed " << ni + 1 << " frames." << std::endl;
    cv::Mat img_left = cv::imread(image_left_vec_path[ni], cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(image_right_vec_path[ni], cv::IMREAD_GRAYSCALE);
    double timestamp = vec_timestamp[ni];
    LOG_IF(FATAL, img_left.empty()) << "Failed to load image at: " << image_left_vec_path[ni];
    LOG(INFO) << "Timestamp " << timestamp;
    system.RunStep(img_left, img_right, timestamp);
    usleep(1e4);
  }
  return 0;
}