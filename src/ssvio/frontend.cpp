//
// Created by weihao on 23-8-9.
//

#include "ssvio/frontend.hpp"
#include "ssvio/frame.hpp"
#include "ssvio/feature.hpp"
#include "ssvio/orbextractor.hpp"
#include "ssvio/mappoint.hpp"
#include "ssvio/keyframe.hpp"
#include "ssvio/map.hpp"
#include "ssvio/backend.hpp"

namespace ssvio {
FrontEnd::FrontEnd()
{
  num_features_init_good_ = Setting::Get<int>("numFeatures.initGood");
  num_features_tracking_good_ = Setting::Get<int>("numFeatures.trackingGood");
  num_features_tracking_bad_ = Setting::Get<int>("numFeatures.trackingBad");
  is_need_undistortion_ = Setting::Get<int>("Camera.NeedUndistortion");
  show_orb_detect_result_ = Setting::Get<int>("View.ORB.Extractor.Result");
  show_lk_result_ = Setting::Get<int>("View.LK.Folw");
  min_init_landmark_ = Setting::Get<int>("Min.Init.Landmark.Num");
  open_backend_optimization_ = Setting::Get<int>("Backend.Open");
  LOG(INFO) << "Open Backend: " << open_backend_optimization_;
}
void FrontEnd::SetCamera(const Camera::Ptr &left, const Camera::Ptr &right)
{
  assert(left != nullptr && right != nullptr);
  left_camera_ = left;
  right_camera_ = right;
}

bool FrontEnd::GrabSteroImage(const cv::Mat &left_img, const cv::Mat &right_img,
                              const double timestamp)
{
  current_frame_.reset(new Frame(left_img, right_img, timestamp));
  /// undistort the images, which is not required in KITTI
  if (is_need_undistortion_)
  {
    left_camera_->UndistortImage(current_frame_->left_image_,
                                 current_frame_->left_image_);
    right_camera_->UndistortImage(current_frame_->right_image_,
                                  current_frame_->right_image_);
  }

  {
    std::unique_lock<std::mutex> lck(map_->mmutex_map_update_);
    switch (track_status_)
    {
    case FrontendStatus::INITING:
      {
        SteroInit();
        break;
      }
    case FrontendStatus::TRACKING_BAD:
    case FrontendStatus::TRACKING_GOOD:
      {
        Track();
        break;
      }
    case FrontendStatus::LOST:
      {
        /// TODO
        break;
      }
    }
  }
  /// 这里的显示是只用于在图像上画二维点，并通过OpenCV进行显示而已，不涉及其他
  if (view_ui_)
  {
    view_ui_->AddCurrentFrame(current_frame_);
  }

  last_frame_ = current_frame_;
  return true;
}

bool FrontEnd::Track()
{
  /// use constant velocity model to preliminarily estimiate the current frame's pose
  if (last_frame_)
  {
    /// T{i k} = T{i-1_i-2} * T{i-1_k}
    /// 更广泛点的说法是 relative_motion_ 表示的是相邻两个帧之间的运动，
    /// 我们有了上一帧的位姿估计则用上一帧的姿态根据恒速模型乘以该运动则表示对当前帧的位姿的初始估计
    current_frame_->SetRelativePose(relative_motion_ * last_frame_->getRelativePose());
  }

  TrackLastFrame();
  int inline_pts = EstimateCurrentPose();

  if (inline_pts > num_features_tracking_good_)
  {
    /// tracking good
    track_status_ = FrontendStatus::TRACKING_GOOD;
  }
  else if (inline_pts > num_features_tracking_bad_)
  {
    /// tracking bad
    track_status_ = FrontendStatus::TRACKING_BAD;
    // LOG(WARNING) << "--inline_pts:-- " << inline_pts;
    // LOG(WARNING) << "----TRACKING BAD!----";
    // LOG(WARNING) << "---------------------";
  }
  else
  {
    /// lost
    track_status_ = FrontendStatus::LOST;
    LOG(WARNING) << "--inline_pts:-- " << inline_pts;
    LOG(WARNING) << "---Tracking LOST!----";
    LOG(WARNING) << "---------------------";
  }

  relative_motion_ =
      current_frame_->getRelativePose() * last_frame_->getRelativePose().inverse();

  /// detect new features; create new mappoints; create new KF
  if (track_status_ == FrontendStatus::TRACKING_BAD)
  {
    DetectFeatures();
    FindFeaturesInRight();
    TriangulateNewPoints();
    InsertKeyFrame();
  }
  return true;
}

int FrontEnd::TrackLastFrame()
{
  std::vector<cv::Point2f> kps_last, kps_current;
  kps_last.reserve(current_frame_->features_left_.size());
  kps_current.reserve(last_frame_->features_left_.size());
  for (size_t i = 0; i < last_frame_->features_left_.size(); i++)
  {
    Feature::Ptr feat = last_frame_->features_left_[i];
    MapPoint::Ptr mappoint = feat->map_point_.lock();
    kps_last.emplace_back(feat->kp_position_.pt);
    if (mappoint)
    {
      Eigen::Vector2d p = left_camera_->world2pixel(mappoint->getPosition(),
                                                    current_frame_->getRelativePose() *
                                                        reference_kf_->getPose());
      kps_current.emplace_back(cv::Point2f(p.x(), p.y()));
    }
    else
    {
      kps_current.emplace_back(feat->kp_position_.pt);
    }
  }
  assert(kps_last.size() > 1 && kps_current.size() > 1);
  std::vector<uchar> status;
  cv::Mat error;
  cv::calcOpticalFlowPyrLK(
      last_frame_->left_image_,
      current_frame_->left_image_,
      kps_last,
      kps_current,
      status,
      error,
      cv::Size(11, 11),
      3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);
  int num_good_track = 0;
  for (size_t i = 0; i < status.size(); i++)
  {
    /// 只会跟踪已经三角化后的点，因为没有三角化的点是无法形成BA约束的
    if (status[i] && !last_frame_->features_left_[i]->map_point_.expired())
    {
      cv::KeyPoint kp(kps_current[i], 7);
      Feature::Ptr feature(new Feature(kp));
      feature->map_point_ = last_frame_->features_left_[i]->map_point_;
      current_frame_->features_left_.emplace_back(feature);
      num_good_track++;
    }
  }
  // LOG(INFO) << "num_good_track: " << num_good_track;
  return num_good_track;
}

int FrontEnd::EstimateCurrentPose()
{
  Eigen::Matrix3d K = left_camera_->getK();

  typedef g2o::BlockSolver_6_3 BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  /// vertes
  VertexPose *vertex_pose = new VertexPose();
  vertex_pose->setId(0);
  /// T{i_k} * T{k_w} = T{i_w}
  vertex_pose->setEstimate(current_frame_->getRelativePose() * reference_kf_->getPose());
  optimizer.addVertex(vertex_pose);

  /// edges
  int index = 1;
  std::vector<EdgeProjectionPoseOnly *> edges;
  std::vector<Feature::Ptr> features;
  features.reserve(current_frame_->features_left_.size());
  edges.reserve(current_frame_->features_left_.size());
  for (size_t i = 0; i < current_frame_->features_left_.size(); i++)
  {
    MapPoint::Ptr mappoint = current_frame_->features_left_[i]->map_point_.lock();
    cv::Point2f pt = current_frame_->features_left_[i]->kp_position_.pt;
    if (mappoint && !mappoint->is_outlier_)
    {
      features.emplace_back(current_frame_->features_left_[i]);
      EdgeProjectionPoseOnly *edge =
          new EdgeProjectionPoseOnly(mappoint->getPosition(), K);
      edge->setId(index);
      edge->setVertex(0, vertex_pose);
      edge->setMeasurement(Eigen::Vector2d(pt.x, pt.y));
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->setRobustKernel(new g2o::RobustKernelHuber);

      edges.emplace_back(edge);
      optimizer.addEdge(edge);
      index++;
    }
  }

  // estimate the Pose and determine the outliers
  // start optimization
  const double chi2_th = 5.991;
  int cnt_outliers = 0;
  int num_iterations = 4;
  for (int iteration = 0; iteration < num_iterations; iteration++)
  {
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cnt_outliers = 0;

    // count the outliers, outlier is not included in estimation until it is regarded as a inlier
    for (size_t i = 0, N = edges.size(); i < N; i++)
    {
      auto e = edges[i];
      if (features[i]->is_outlier_)
      {
        e->computeError();
      }
      if (e->chi2() > chi2_th)
      {
        features[i]->is_outlier_ = true;
        e->setLevel(1);
        cnt_outliers++;
      }
      else
      {
        features[i]->is_outlier_ = false;
        e->setLevel(0);
      }

      // remove the robust kernel to see if it's outlier
      if (iteration == num_iterations - 2)
      {
        e->setRobustKernel(nullptr);
      }
    }
  }

  /// set pose
  current_frame_->SetPose(vertex_pose->estimate());
  /// T{i_k} = T{i_w} * T{k_w}.inverse()
  current_frame_->SetRelativePose(vertex_pose->estimate() *
                                  reference_kf_->getPose().inverse());

  for (auto &feat : features)
  {
    if (feat->is_outlier_)
    {
      MapPoint::Ptr mp = feat->map_point_.lock();
      if (mp && current_frame_->frame_id_ - reference_kf_->frame_id_ <= 2)
      {
        mp->is_outlier_ = true;
        map_->AddOutlierMapPoint(mp->id_);
      }
      feat->map_point_.reset();
      feat->is_outlier_ = false;
    }
  }

  //  LOG(INFO) << "Frontend: Outliers/Inliers in frontend current pose estimating: "
  // << cntOutliers << "/" << features.size() - cntOutliers;

  return features.size() - cnt_outliers;
}

int FrontEnd::DetectFeatures()
{
  cv::Mat mask(current_frame_->left_image_.size(), CV_8UC1, cv::Scalar::all(255));
  for (const auto &feat : current_frame_->features_left_)
  {
    cv::rectangle(mask,
                  feat->kp_position_.pt - cv::Point2f(10, 10),
                  feat->kp_position_.pt + cv::Point2f(10, 10),
                  0,
                  cv::FILLED);
  }

  std::vector<cv::KeyPoint> kps;
  if (track_status_ == FrontendStatus::INITING)
  {
    orb_extractor_init_->Detect(current_frame_->left_image_, mask, kps);
  }
  else
  {
    orb_extractor_->Detect(current_frame_->left_image_, mask, kps);
  }
  int cnt_detected = 0;
  for (const auto &kp : kps)
  {
    Feature::Ptr feature(new Feature(kp));
    current_frame_->features_left_.emplace_back(feature);
    cnt_detected++;
  }
  if (show_orb_detect_result_ && cnt_detected > 0)
  {
    cv::Mat show_img(current_frame_->left_image_.size(), CV_8UC1);
    cv::drawKeypoints(current_frame_->left_image_, kps, show_img, cv::Scalar(0, 255, 0));
    cv::imshow("orb_detect_result", show_img);
    cv::waitKey(1);
  }

  if (track_status_ == FrontendStatus::TRACKING_BAD)
  {
    // LOG(INFO) << "Detect New Features: " << cnt_detected
    //           << " left image features size: " << current_frame_->features_left_.size();
  }
  return cnt_detected;
}

int FrontEnd::FindFeaturesInRight()
{
  std::vector<cv::Point2f> left_cam_kps, right_cam_kps;
  left_cam_kps.reserve(current_frame_->features_left_.size());
  right_cam_kps.reserve(current_frame_->features_left_.size());
  for (const auto &feat : current_frame_->features_left_)
  {
    left_cam_kps.push_back(feat->kp_position_.pt);
    auto pt_in_map = feat->map_point_.lock();
    if (pt_in_map)
    {
      /// 如果是已经被三角化后的点,则通过第一个相机的位姿和相机之间的外参投影到第二个相机的平面上去
      Eigen::Vector2d p = right_camera_->world2pixel(pt_in_map->getPosition(),
                                                     current_frame_->getRelativePose() *
                                                         reference_kf_->getPose());
      right_cam_kps.push_back(cv::Point2f(p.x(), p.y()));
    }
    else
    {
      right_cam_kps.push_back(feat->kp_position_.pt);
    }
  }

  // LK flow: calculate the keypoints' positions in right frame
  std::vector<uchar> status;
  cv::Mat error;
  cv::calcOpticalFlowPyrLK(
      current_frame_->left_image_,
      current_frame_->right_image_,
      left_cam_kps,
      right_cam_kps,
      status,
      error,
      cv::Size(11, 11),
      3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  /// create new feature objects in right frame
  int num_good_points = 0;
  current_frame_->features_right_.reserve(status.size());
  for (size_t i = 0; i < status.size(); ++i)
  {
    if (status[i])
    {
      /// only the position of keypoint is needed, so size 7 is just for creation with no meaning
      cv::KeyPoint kp(right_cam_kps[i], 7);
      Feature::Ptr feat(new Feature(kp));
      feat->is_on_left_frame_ = false;
      current_frame_->features_right_.push_back(feat);
      num_good_points++;
    }
    else
    {
      current_frame_->features_right_.push_back(nullptr);
    }
  }

  if (show_lk_result_)
  {
    cv::Mat show;
    cv::cvtColor(current_frame_->left_image_, show, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < current_frame_->features_left_.size(); i++)
    {
      if (status[i])
      {
        cv::Point2i pt1 = current_frame_->features_left_[i]->kp_position_.pt;
        cv::Point2i pt2 = current_frame_->features_right_[i]->kp_position_.pt;
        cv::circle(show, pt1, 2, cv::Scalar(0, 250, 0), 2);
        cv::line(show, pt1, pt2, cv::Scalar(0, 0, 255), 1.5);
      }
    }
    cv::imshow("LK", show);
    cv::waitKey(1);
  }
  if (track_status_ == FrontendStatus::TRACKING_BAD)
  {
    // LOG(INFO) << "LK Track New Features: " << num_good_points;
  }    
  return num_good_points;
}

bool FrontEnd::SteroInit()
{
  int cnt_detected_features = DetectFeatures();
  int cnt_track_features = FindFeaturesInRight();
  if (cnt_track_features < num_features_init_good_)
  {
    LOG(WARNING) << "Too few feature points";
    return false;
  }
  bool build_init_map_success = BuidInitMap();
  if (build_init_map_success)
  {
    track_status_ = FrontendStatus::TRACKING_GOOD;
    return true;
  }
  return false;
}

bool FrontEnd::BuidInitMap()
{
  std::vector<Sophus::SE3d> poses{left_camera_->getPose(), right_camera_->getPose()};
  size_t cnt_init_landmarks = 0;
  for (size_t i = 0, N = current_frame_->features_left_.size(); i < N; i++)
  {
    if (current_frame_->features_right_[i] == nullptr)
      continue;
    /// create mappoints by triangulation
    std::vector<Eigen::Vector3d> points{
        left_camera_->pixel2camera(
            Eigen::Vector2d(current_frame_->features_left_[i]->kp_position_.pt.x,
                            current_frame_->features_left_[i]->kp_position_.pt.y)),
        right_camera_->pixel2camera(
            Eigen::Vector2d(current_frame_->features_right_[i]->kp_position_.pt.x,
                            current_frame_->features_right_[i]->kp_position_.pt.y))};
    Eigen::Vector3d pworld = Eigen::Vector3d::Zero();

    if (triangulation(poses, points, pworld) && pworld[2] > 0)
    {
      /// if successfully triangulate, then create new mappoint and insert it to the map
      MapPoint::Ptr new_map_point(new MapPoint);
      new_map_point->SetPosition(pworld);
      current_frame_->features_left_[i]->map_point_ = new_map_point;
      current_frame_->features_right_[i]->map_point_ = new_map_point;

      if (map_)
        map_->InsertMapPoint(new_map_point);
      if (view_ui_)
        view_ui_->AddShowPointCloud(new_map_point->getPosition());

      cnt_init_landmarks++;
    }
  }

  if (cnt_init_landmarks < min_init_landmark_)
  {
    LOG(WARNING) << "Build init map Failed, have " << cnt_init_landmarks
                 << " points, min is: " << min_init_landmark_;
    return false;
  }

  InsertKeyFrame();
  LOG(INFO) << "Build init map success, have " << cnt_init_landmarks << " points";
  cv::destroyAllWindows();
  return true;
}

int FrontEnd::TriangulateNewPoints()
{
  auto cv_point2f_to_vec2 = [](cv::Point2f &pt) { return Eigen::Vector2d(pt.x, pt.y); };
  std::vector<Sophus::SE3d> poses{left_camera_->getPose(), right_camera_->getPose()};
  Sophus::SE3d current_pose_Twc =
      (current_frame_->getRelativePose() * reference_kf_->getPose()).inverse();
  size_t cnt_trangulat_pts = 0, cnt_previous_mappoint = 0;
  for (size_t i = 0; i < current_frame_->features_left_.size(); i++)
  {
    Feature::Ptr feat_left = current_frame_->features_left_[i];
    Feature::Ptr feat_right = current_frame_->features_right_[i];
    MapPoint::Ptr mp = feat_left->map_point_.lock();
    if (!current_frame_->features_left_[i]->map_point_.expired())
    {
      /// no need to triangulate
      cnt_previous_mappoint++;
      continue;
    }
    /// LK track failed
    if (feat_right == nullptr)
      continue;

    std::vector<Eigen::Vector3d> points;
    points.emplace_back(
        left_camera_->pixel2camera(cv_point2f_to_vec2(feat_left->kp_position_.pt)));
    points.emplace_back(
        right_camera_->pixel2camera(cv_point2f_to_vec2(feat_right->kp_position_.pt)));
    Eigen::Vector3d pt_camera1;
    if (triangulation(poses, points, pt_camera1) && pt_camera1[2] > 0)
    {
      MapPoint::Ptr new_mappoint(new MapPoint);
      new_mappoint->SetPosition(current_pose_Twc * pt_camera1);
      current_frame_->features_left_[i]->map_point_ = new_mappoint;
      current_frame_->features_right_[i]->map_point_ = new_mappoint;
      if (map_)
        map_->InsertMapPoint(new_mappoint);
      if (view_ui_)
        view_ui_->AddShowPointCloud(new_mappoint->getPosition());
      cnt_trangulat_pts++;
    }
  }

  // LOG(INFO) << "Triangularte new points size: " << cnt_trangulat_pts;
  return cnt_trangulat_pts;
}

bool FrontEnd::InsertKeyFrame()
{
  Eigen::Matrix<double, 6, 1> se3_zero = Eigen::Matrix<double, 6, 1>::Zero();
  KeyFrame::Ptr new_keyframe = KeyFrame::CreateKF(current_frame_);
  if (track_status_ == FrontendStatus::INITING)
  {
    new_keyframe->SetPose(Sophus::SE3d::exp(se3_zero));
  }
  else
  {
    /// current_frame_->pose(); T_ik * T_kw = T_iw
    new_keyframe->SetPose(current_frame_->getRelativePose() * reference_kf_->getPose());
    new_keyframe->last_key_frame_ = reference_kf_;
    new_keyframe->relative_pose_to_last_KF_ = current_frame_->getRelativePose(); /// T_ik
  }

  //////////////////////////////////////////////////////////////
  if (backend_)
  {
    backend_->InsertKeyFrame(new_keyframe, open_backend_optimization_);
  }
  //////////////////////////////////////////////////////////////

  reference_kf_ = new_keyframe;

  current_frame_->SetRelativePose(Sophus::SE3d::exp(se3_zero));

  return true;
}

void FrontEnd::SetOrbExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb)
{
  assert(orb != nullptr);
  orb_extractor_ = orb;
}

void FrontEnd::SetViewUI(const std::shared_ptr<ui::PangolinWindow> &ui)
{
  assert(ui != nullptr);
  view_ui_ = ui;
}

void FrontEnd::SetOrbInitExtractor(const std::shared_ptr<ssvio::ORBextractor> &orb)
{
  assert(orb != nullptr);
  orb_extractor_init_ = orb;
}

void FrontEnd::SetMap(const shared_ptr<Map> &map)
{
  assert(map != nullptr);
  map_ = map;
}

void FrontEnd::SetBackend(const shared_ptr<Backend> &backend)
{
  assert(backend != nullptr);
  backend_ = backend;
}

} // namespace ssvio
