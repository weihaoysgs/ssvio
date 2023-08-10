#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <condition_variable>
#include <opencv2/core/core.hpp>
#include <librealsense2/rs.hpp>
#include "librealsense2/rsutil.h"
#include "unistd.h"
#include "opencv2/opencv.hpp"
#include "ssvio/system.hpp"

using namespace std;
bool b_continue_session;

void exit_loop_handler(int s)
{
  cout << "Finishing session" << endl;
  b_continue_session = false;
}

rs2_vector interpolateMeasure(const double target_time, const rs2_vector current_data,
                              const double current_time, const rs2_vector prev_data,
                              const double prev_time);

static rs2_option get_sensor_option(const rs2::sensor &sensor)
{
  // Sensors usually have several options to control their properties
  //  such as Exposure, Brightness etc.

  std::cout << "Sensor supports the following options:\n" << std::endl;

  // The following loop shows how to iterate over all available options
  // Starting from 0 until RS2_OPTION_COUNT (exclusive)
  for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
  {
    rs2_option option_type = static_cast<rs2_option>(i);
    //SDK enum types can be streamed to get a string that represents them
    std::cout << "  " << i << ": " << option_type;

    // To control an option, use the following api:

    // First, verify that the sensor actually supports this option
    if (sensor.supports(option_type))
    {
      std::cout << std::endl;

      // Get a human readable description of the option
      const char *description = sensor.get_option_description(option_type);
      std::cout << "       Description   : " << description << std::endl;

      // Get the current value of the option
      float current_value = sensor.get_option(option_type);
      std::cout << "       Current Value : " << current_value << std::endl;

      //To change the value of an option, please follow the change_sensor_option() function
    }
    else
    {
      std::cout << " is not supported" << std::endl;
    }
  }

  uint32_t selected_sensor_option = 0;
  return static_cast<rs2_option>(selected_sensor_option);
}

int main(int argc, char **argv)
{
  string file_name;

  struct sigaction sigIntHandler;

  sigIntHandler.sa_handler = exit_loop_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, NULL);
  b_continue_session = true;

  double offset = 0; // ms

  rs2::context ctx;
  rs2::device_list devices = ctx.query_devices();
  rs2::device selected_device;
  if (devices.size() == 0)
  {
    std::cerr << "No device connected, please connect a RealSense device" << std::endl;
    return 0;
  }
  else
    selected_device = devices[0];

  std::vector<rs2::sensor> sensors = selected_device.query_sensors();
  int index = 0;
  // We can now iterate the sensors and print their names

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;
  // Create a configuration for configuring the pipeline with a non default profile
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
  cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);

  // IMU callback
  std::mutex imu_mutex;
  std::condition_variable cond_image_rec;

  cv::Mat imCV, imRightCV;
  int width_img, height_img;
  double timestamp_image = -1.0;
  bool image_ready = false;
  int count_im_buffer = 0; // count dropped frames

  auto imu_callback = [&](const rs2::frame &frame) {
    std::unique_lock<std::mutex> lock(imu_mutex);

    if (rs2::frameset fs = frame.as<rs2::frameset>())
    {
      count_im_buffer++;

      double new_timestamp_image = fs.get_timestamp() * 1e-3;
      if (abs(timestamp_image - new_timestamp_image) < 0.001)
      {
        // cout << "Two frames with the same timeStamp!!!\n";
        count_im_buffer--;
        return;
      }

      rs2::video_frame ir_frameL = fs.get_infrared_frame(1);
      rs2::video_frame ir_frameR = fs.get_infrared_frame(2);

      imCV = cv::Mat(cv::Size(width_img, height_img),
                     CV_8U,
                     (void *)(ir_frameL.get_data()),
                     cv::Mat::AUTO_STEP);
      imRightCV = cv::Mat(cv::Size(width_img, height_img),
                          CV_8U,
                          (void *)(ir_frameR.get_data()),
                          cv::Mat::AUTO_STEP);

      timestamp_image = fs.get_timestamp() * 1e-3;
      image_ready = true;

      lock.unlock();
      cond_image_rec.notify_all();
    }
  };

  rs2::pipeline_profile pipe_profile = pipe.start(cfg, imu_callback);

  rs2::stream_profile cam_left = pipe_profile.get_stream(RS2_STREAM_INFRARED, 1);
  rs2::stream_profile cam_right = pipe_profile.get_stream(RS2_STREAM_INFRARED, 2);

  float *Rlr = cam_right.get_extrinsics_to(cam_left).rotation;
  float *tlr = cam_right.get_extrinsics_to(cam_left).translation;
  std::cout << "Tlr  = " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
      std::cout << Rlr[i * 3 + j] << ", ";
    std::cout << tlr[i] << "\n";
  }

  rs2_intrinsics intrinsics_left =
      cam_left.as<rs2::video_stream_profile>().get_intrinsics();
  width_img = intrinsics_left.width;
  height_img = intrinsics_left.height;
  cout << "Left camera: \n";
  std::cout << " fx = " << intrinsics_left.fx << std::endl;
  std::cout << " fy = " << intrinsics_left.fy << std::endl;
  std::cout << " cx = " << intrinsics_left.ppx << std::endl;
  std::cout << " cy = " << intrinsics_left.ppy << std::endl;
  std::cout << " height = " << intrinsics_left.height << std::endl;
  std::cout << " width = " << intrinsics_left.width << std::endl;
  std::cout << " Coeff = " << intrinsics_left.coeffs[0] << ", "
            << intrinsics_left.coeffs[1] << ", " << intrinsics_left.coeffs[2] << ", "
            << intrinsics_left.coeffs[3] << ", " << intrinsics_left.coeffs[4] << ", "
            << std::endl;
  std::cout << " Model = " << intrinsics_left.model << std::endl;

  rs2_intrinsics intrinsics_right =
      cam_right.as<rs2::video_stream_profile>().get_intrinsics();
  width_img = intrinsics_right.width;
  height_img = intrinsics_right.height;
  cout << "Right camera: \n";
  std::cout << " fx = " << intrinsics_right.fx << std::endl;
  std::cout << " fy = " << intrinsics_right.fy << std::endl;
  std::cout << " cx = " << intrinsics_right.ppx << std::endl;
  std::cout << " cy = " << intrinsics_right.ppy << std::endl;
  std::cout << " height = " << intrinsics_right.height << std::endl;
  std::cout << " width = " << intrinsics_right.width << std::endl;
  std::cout << " Coeff = " << intrinsics_right.coeffs[0] << ", "
            << intrinsics_right.coeffs[1] << ", " << intrinsics_right.coeffs[2] << ", "
            << intrinsics_right.coeffs[3] << ", " << intrinsics_right.coeffs[4] << ", "
            << std::endl;
  std::cout << " Model = " << intrinsics_right.model << std::endl;

  double timestamp;
  cv::Mat im, imRight;

  double t_resize = 0.f;
  double t_track = 0.f;
  const std::string config_file_path =
      "/home/weihao/codespace/ssvio/config/kitti_00.yaml";
  ssvio::System system(config_file_path);
  while (!system.getViewUi()->ShouldQuit())
  {
    std::vector<rs2_vector> vGyro;
    std::vector<double> vGyro_times;
    std::vector<rs2_vector> vAccel;
    std::vector<double> vAccel_times;

    {
      std::unique_lock<std::mutex> lk(imu_mutex);
      if (!image_ready)
        cond_image_rec.wait(lk);

      if (count_im_buffer > 1)
        cout << count_im_buffer - 1 << " dropped frs\n";
      count_im_buffer = 0;

      timestamp = timestamp_image;
      im = imCV.clone();
      imRight = imRightCV.clone();
      system.RunStep(im, imRightCV, timestamp);

      image_ready = false;
    }
  }
}