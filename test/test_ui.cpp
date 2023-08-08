//
// Created by weihao on 23-8-7.
//
#include "ui/pangolin_window.hpp"
#include "iostream"
#include "unistd.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "filesystem"
#include "fstream"

DEFINE_string(kitti_dataset_path,
              "/home/weihao/dataset/kitti/data_odometry_gray/dataset/sequences/00",
              "kitti dataset path");
using namespace std;

void LoadImages(const string &str_path_to_sequence,
                vector<string> &str_image_left_vec_path,
                vector<string> &str_image_right_vec_path,
                vector<double> &timestamps_vec);

int main(int argc, char **argv)
{
  google::InitGoogleLogging("test_ui");
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  ui::PangolinWindow ui;
  LOG_ASSERT(ui.Init()) << "Ui init failed";

  std::string str_sequence_path = fLS::FLAGS_kitti_dataset_path;

  // load sequence frames
  std::vector<std::string> image_left_vec_path, image_right_vec_path;
  std::vector<double> vec_timestamp;
  LoadImages(str_sequence_path, image_left_vec_path, image_right_vec_path, vec_timestamp);
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

    LOG(INFO) << "Process time: " << timestamp << " image";
    usleep(1e4);
  }
  return 0;
}

/// for KITTI gray database
void LoadImages(const string &str_path_to_sequence,
                vector<string> &str_image_left_vec_path,
                vector<string> &str_image_right_vec_path,
                vector<double> &timestamps_vec)
{
  string strPathTimeFile = str_path_to_sequence + "/times.txt";

  std::ifstream fTimes(strPathTimeFile, ios::in | ios::app);

  if (!fTimes.is_open())
    LOG(FATAL) << "Open Failed";
  while (!fTimes.eof())
  {
    string s;
    getline(fTimes, s);
    if (!s.empty())
    {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      timestamps_vec.push_back(t);
    }
    else
    {
      LOG(ERROR) << "Empty";
    }
  }

  string strPrefixLeft = str_path_to_sequence + "/image_0/";
  string strPrefixRight = str_path_to_sequence + "/image_1/";

  const size_t nTimes = timestamps_vec.size();
  str_image_left_vec_path.resize(nTimes);
  str_image_right_vec_path.resize(nTimes);

  for (int i = 0; i < nTimes; i++)
  {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    str_image_left_vec_path[i] = strPrefixLeft + ss.str() + ".png";
    str_image_right_vec_path[i] = strPrefixRight + ss.str() + ".png";
  }
  fTimes.close();
}
