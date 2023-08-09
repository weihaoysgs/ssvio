//
// Created by weihao on 23-8-9.
//
#include "ssvio/system.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"

DEFINE_string(config_yaml_path, "/home/weihao/codespace/ssvio/config/kitti_00.yaml", "System config file path");

int main(int argc, char **argv)
{
  google::InitGoogleLogging("test_system");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags(&argc, &argv, true);
  ssvio::System system(fLS::FLAGS_config_yaml_path);
  while (!system.getViewUi()->ShouldQuit())
  {
  }
  return 0;
}