//
// Created by weihao on 23-8-10.
//

#ifndef SSVIO_BACKEND_HPP
#define SSVIO_BACKEND_HPP

#include "Eigen/Core"
#include "Eigen/Dense"
#include "ssvio/g2otypes.hpp"
#include "ui/pangolin_window.hpp"

namespace ssvio {

class Camera;
class Map;
class Frame;
class KeyFrame;
class LoopClosing;

class Backend
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<Backend> Ptr;
  Backend();

  void SetViewer(std::shared_ptr<ui::PangolinWindow> viewer) { view_ui_ = viewer; }
  void SetLoopClosing(std::shared_ptr<LoopClosing> lp) { loop_closing_ = lp; }
  void SetMap(std::shared_ptr<Map> map) { map_ = map; }
  void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right);


  void Stop();
  void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF, const bool optimization);
  void RequestPause();
  bool IfHasPaused();
  void Resume();
  bool CheckNewKeyFrames();
  void ProcessNewKeyFrame();
  void BackendLoop();
  void OptimizeActiveMap();

 public:
  std::mutex _mmutex_new_KF_;
  std::mutex _mmutex_stop_;
  std::atomic<bool> _backend_is_running_;
  std::atomic<bool> _request_pause_;
  std::atomic<bool> _has_paused_;
  bool need_optimization_ = false;

 private:
  std::shared_ptr<Camera> camera_left_ = nullptr, camera_right_ = nullptr;
  std::shared_ptr<ui::PangolinWindow> view_ui_ = nullptr;
  std::shared_ptr<LoopClosing> loop_closing_ = nullptr;

  std::list<std::shared_ptr<KeyFrame>> new_keyfrmaes_;
  std::shared_ptr<KeyFrame> current_keyframe_ = nullptr;
  std::shared_ptr<Map> map_ = nullptr;
  std::thread backend_thread_;
};

} // namespace ssvio

#endif //SSVIO_BACKEND_HPP
