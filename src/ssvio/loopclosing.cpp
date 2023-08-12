//
// Created by weihao on 23-8-10.
//

#include "ssvio/loopclosing.hpp"
#include "ssvio/keyframe.hpp"
#include "ssvio/setting.hpp"
#include "ssvio/mappoint.hpp"
#include "ssvio/feature.hpp"
#include "common/couttools.hpp"
#include "ssvio/camera.hpp"
#include "ssvio/g2otypes.hpp"

namespace ssvio {

LoopClosing::LoopClosing()
{
  LoadParam();
  GenerateORBextractor();
  loop_thread_is_running_.store(true);

  cv_matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");

  std::string orb_voc_path = Setting::Get<std::string>("DBOW2.VOC.Path");
  LOG_ASSERT(orb_voc_path.empty() == false);
  LOG(INFO) << "orb_voc_path: " << orb_voc_path << orb_voc_path;

  if (open_loop_closing_)
  {
    dbow2_vocabulary_ = std::make_unique<ORBVocabulary>();
    LOG(WARNING) << "Reading ORB Vocabulary txt File, Wait a Second.....";
    dbow2_vocabulary_->loadFromTextFile(orb_voc_path);
    loop_closing_thread_ = std::thread(&LoopClosing::LoopClosingThread, this);
  }
}

void LoopClosing::LoopClosingThread()
{
  while (loop_thread_is_running_.load())
  {
    if (CheckNewKeyFrame())
    {
      ProcessNewKeyframe();

      bool confirm_loop_closing = false;
      if (key_frame_database_.size() > keyframe_database_min_size_)
      {
        if (DetectLoop())
        {
          if (MatchFeatures())
          {
            confirm_loop_closing = ComputeCorrectPose();
            if (confirm_loop_closing)
            {
              LoopCorrect();
            }
          }
        }
      }
      if (!confirm_loop_closing)
      {
        AddToKeyframeDatabase();
      }
    }

    usleep(1000);
  }
}

bool LoopClosing::DetectLoop()
{
  float max_score = 0.0;
  unsigned long best_match_id = 0;

  for (const auto &db : key_frame_database_)
  {
    if (current_keyframe_->key_frame_id_ - db.second->key_frame_id_ < 20)
      break;
    if (!current_keyframe_->bow2_vec_.empty() && !db.second->bow2_vec_.empty())
    {
      float similarity_score =
          dbow2_vocabulary_->score(current_keyframe_->bow2_vec_, db.second->bow2_vec_);
      if (similarity_score > max_score)
      {
        max_score = similarity_score;
        best_match_id = db.first;
      }
    }
  }

  if (max_score < loop_threshold_heigher_)
  {
    return false;
  }

  loop_keyframe_ = key_frame_database_.at(best_match_id);
  LOG(INFO) << "----------------------------------------------------" << TAIL;
  LOG(INFO) << GREEN << "LoopClosing: find potential Candidate KF. Score is " << max_score
            << TAIL;
  LOG(INFO) << GREEN << "Keyframe Database size: " << key_frame_database_.size() << TAIL;
  return true;

  return false;
}

bool LoopClosing::MatchFeatures()
{
  std::vector<cv::DMatch> matches;
  cv_matcher_->match(
      loop_keyframe_->ORBDescriptors_, current_keyframe_->ORBDescriptors_, matches);

  auto min_max = std::minmax_element(
      matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) {
        return m1.distance < m2.distance;
      });
  double min_distance = min_max.first->distance;

  set_valid_feature_matches_.clear();
  for (const auto &mt : matches)
  {
    if (mt.distance <= std::max(2 * min_distance, 30.0))
    {
      /// class_id 表示是第几个特征点，去掉了金字塔,同一个金字塔层级的 class_id 是一样的
      int loop_feature_id = loop_keyframe_->pyramid_key_points_[mt.queryIdx].class_id;
      int current_feature_id =
          current_keyframe_->pyramid_key_points_[mt.trainIdx].class_id;

      // the matches of keypoints belonging to the same feature pair shouldn't be inserted into the valid matches twice
      if (set_valid_feature_matches_.find({current_feature_id, loop_feature_id}) !=
          set_valid_feature_matches_.end())
      {
        continue;
      }
      set_valid_feature_matches_.insert({current_feature_id, loop_feature_id});
    }
  }
  LOG(INFO) << GREEN << "LoopClosing: number of valid feature matches: "
            << set_valid_feature_matches_.size() << TAIL;

  if (set_valid_feature_matches_.size() < 10)
  {
    return false;
  }

  return true;
}

bool LoopClosing::ComputeCorrectPose()
{
  std::vector<cv::Point3f> loop_point3f;
  std::vector<cv::Point2f> current_point2f;
  std::vector<cv::DMatch> matches_with_mappoint;

  for (auto iter = set_valid_feature_matches_.begin();
       iter != set_valid_feature_matches_.end();)
  {
    int loop_feature_id = iter->second;
    int current_feature_id = iter->first;
    MapPoint::Ptr mp = loop_keyframe_->features_left_[loop_feature_id]->map_point_.lock();
    if (mp)
    {
      Eigen::Vector3d p = mp->getPosition();
      current_point2f.push_back(
          current_keyframe_->features_left_[current_feature_id]->kp_position_.pt);
      loop_point3f.push_back(cv::Point3f(p(0), p(1), p(2)));
      // useful if needs to draw the matches
      cv::DMatch valid_match(current_feature_id, loop_feature_id, 10.0);
      matches_with_mappoint.push_back(valid_match);
      iter++;
    }
    else
    {
      iter = set_valid_feature_matches_.erase(iter);
    }
  }

  LOG(INFO) << GREEN << "LoopClosing: number of valid matches with mappoints:"
            << loop_point3f.size() << TAIL;
  if (show_loop_closing_result_)
  {
    //  show the match result
    cv::Mat img_goodmatch;
    cv::drawMatches(current_keyframe_->image_left_,
                    current_keyframe_->GetKeyPoints(),
                    loop_keyframe_->image_left_,
                    loop_keyframe_->GetKeyPoints(),
                    matches_with_mappoint,
                    img_goodmatch);
    cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.5, 0.5);
    cv::imshow("valid matches with mappoints", img_goodmatch);
    cv::waitKey(1);
  }

  if (loop_point3f.size() < 10)
    return false;

  cv::Mat rvec, tvec, R, K;
  cv::eigen2cv(left_camera_->getK(), K);
  Eigen::Matrix3d Reigen;
  Eigen::Vector3d teigen;
  // use "try - catch" since cv::solvePnPRansac may fail because of terrible match result
  // and I don't know why the result of solvePnPRansac() is sometimes not reliable
  //      even the reprojection error of inlier is high
  try
  {
    cv::solvePnPRansac(
        loop_point3f, current_point2f, K, cv::Mat(), rvec, tvec, false, 100, 5.991, 0.99);
  } catch (...)
  {
    return false;
  }

  cv::Rodrigues(rvec, R);
  cv::cv2eigen(R, Reigen);
  cv::cv2eigen(tvec, teigen);

  corrected_current_pose_ = Sophus::SE3d(Reigen, teigen);

  int cnt_inliner = OptimizeCurrentPose();
  if (cnt_inliner < 10)
    return false;
  LOG(INFO) << "LoopClosing: number of match inliers (after optimization): " <<  cnt_inliner;

  return false;
}

int LoopClosing::OptimizeCurrentPose()
{
  auto to_vec2 = [](cv::Point2f &pt) { return Eigen::Vector2d(pt.x, pt.y); };
  typedef g2o::BlockSolver_6_3 BlockerSolverType;
  typedef g2o::LinearSolverDense<BlockerSolverType::PoseMatrixType> LinerrSolverType;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockerSolverType>(g2o::make_unique<LinerrSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  VertexPose *vertex_pose = new VertexPose();
  vertex_pose->setId(0);
  vertex_pose->setEstimate(current_keyframe_->getPose());

  optimizer.addVertex(vertex_pose);

  int index = 1;
  Eigen::Matrix3d left_cam_K = left_camera_->getK();
  Eigen::Matrix3d right_cam_K = right_camera_->getK();

  std::vector<EdgeProjectionPoseOnly *> edges;
  edges.reserve(set_valid_feature_matches_.size());
  std::vector<bool> edge_is_outlier;
  edge_is_outlier.reserve(set_valid_feature_matches_.size());
  std::vector<std::pair<int, int>> matches;
  matches.reserve(set_valid_feature_matches_.size());

  for (auto &match : set_valid_feature_matches_)
  {
    int current_feature_id = match.first;
    int loop_feature_id = match.second;
    Feature::Ptr feat = current_keyframe_->features_left_[current_feature_id];
    Eigen::Vector2d observe = to_vec2(feat->kp_position_.pt);
    Eigen::Vector3d pt3d =
        loop_keyframe_->features_left_[loop_feature_id]->map_point_.lock()->getPosition();

    EdgeProjectionPoseOnly *edge = nullptr;
    if (feat->is_on_left_frame_)
      edge = new EdgeProjectionPoseOnly(pt3d, left_cam_K);
    else
      edge = new EdgeProjectionPoseOnly(pt3d, right_cam_K);

    edge->setId(index);
    edge->setVertex(0, vertex_pose);
    edge->setInformation(Eigen::Matrix2d::Identity());
    edge->setMeasurement(observe);
    edge->setRobustKernel(new g2o::RobustKernelHuber);
    optimizer.addEdge(edge);

    edges.emplace_back(edge);
    edge_is_outlier.emplace_back(false);
    matches.emplace_back(match);
    index++;
  }

  // estimate the Pose and determine the outliers
  // start optimization
  const double chi2_th = 5.991;
  int cnt_outliers = 0;
  int num_iterations = 4;

  optimizer.initializeOptimization();
  optimizer.optimize(10);

  for (int iter = 0; iter < num_iterations; iter++)
  {
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cnt_outliers = 0;
    for (size_t i = 0; i < edges.size(); i++)
    {
      auto edge = edges[i];
      if (edge_is_outlier[i])
      {
        edge->computeError();
      }
      if (edge->chi2() > chi2_th)
      {
        edge_is_outlier[i] = true;
        edge->setLevel(1);
        cnt_outliers++;
      }
      else
      {
        edge->setLevel(0);
        edge_is_outlier[i] = false;
      }
      if (iter == num_iterations - 2)
      {
        edge->setRobustKernel(nullptr);
      }
    }
  }

  for (size_t i = 0; i < edge_is_outlier.size(); i++)
  {
    if (edge_is_outlier[i])
    {
      set_valid_feature_matches_.erase(matches[i]);
    }
  }

  corrected_current_pose_ = vertex_pose->estimate();
  return set_valid_feature_matches_.size();
}

void LoopClosing::LoopCorrect()
{
  return;
}

void LoopClosing::ProcessNewKeyframe()
{
  {
    std::unique_lock<std::mutex> lck(all_new_keyframe_mutex_);
    current_keyframe_ = all_new_keyframes_.front();

    all_new_keyframes_.pop_front();
  }

  std::vector<cv::KeyPoint> pyramid_points;
  pyramid_points.reserve(pyramid_level_num_ * current_keyframe_->features_left_.size());
  for (size_t i = 0; i < current_keyframe_->features_left_.size(); i++)
  {
    /// class id 标记的是序号
    current_keyframe_->features_left_[i]->kp_position_.class_id = i;
    for (int level = 0; level < pyramid_level_num_; level++)
    {
      cv::KeyPoint kp(current_keyframe_->features_left_[i]->kp_position_);
      kp.octave = level;
      kp.response = -1;
      kp.class_id = i;
      pyramid_points.emplace_back(kp);
    }
  }
  // remove the pyramid keypoints which are not FAST corner or beyond borders
  // compute their orientations and sizes
  orb_extractor_->ScreenAndComputeKPsParams(current_keyframe_->image_left_,
                                            pyramid_points,
                                            current_keyframe_->pyramid_key_points_);

  // calculate the orb descriptors of all valid pyramid keypoints
  orb_extractor_->CalcDescriptors(current_keyframe_->image_left_,
                                  current_keyframe_->pyramid_key_points_,
                                  current_keyframe_->ORBDescriptors_);
  std::vector<cv::Mat> desc =
      ConvertToDescriptorVector(current_keyframe_->ORBDescriptors_);

  dbow2_vocabulary_->transform(desc, current_keyframe_->bow2_vec_);
}

void LoopClosing::StopLoopClosing()
{
  while (CheckNewKeyFrame())
  {
    usleep(1e5);
  }
  loop_thread_is_running_.store(false);
  loop_closing_thread_.join();
}

void LoopClosing::AddToKeyframeDatabase()
{
  key_frame_database_.insert({current_keyframe_->key_frame_id_, current_keyframe_});
}

bool LoopClosing::CheckNewKeyFrame()
{
  std::unique_lock<std::mutex> lck(all_new_keyframe_mutex_);
  return (!all_new_keyframes_.empty());
}

void LoopClosing::InsertNewKeyFrame(const std::shared_ptr<KeyFrame> new_kf)
{
  std::unique_lock<std::mutex> lck(all_new_keyframe_mutex_);
  if (last_closed_keyframe_ == nullptr ||
      new_kf->key_frame_id_ - last_closed_keyframe_->key_frame_id_ > 5)
  {
    all_new_keyframes_.push_back(new_kf);
  }
  else
  {
    // TODO
  }
}

std::vector<cv::Mat> LoopClosing::ConvertToDescriptorVector(const cv::Mat &descriptors)
{
  assert(descriptors.rows > 0);
  std::vector<cv::Mat> desc;
  desc.reserve(descriptors.rows);
  for (int j = 0; j < descriptors.rows; j++)
    desc.push_back(descriptors.row(j));
  return desc;
}

void LoopClosing::SetSteroCamera(std::shared_ptr<Camera> left,
                                 std::shared_ptr<Camera> right)
{
  assert(left != nullptr && right != nullptr);
  left_camera_ = left;
  right_camera_ = right;
}

void LoopClosing::GenerateORBextractor()
{
  int num_orb_bew_features = Setting::Get<int>("ORBextractor.nNewFeatures");
  float scale_factor = Setting::Get<float>("ORBextractor.scaleFactor");
  int n_levels = Setting::Get<int>("ORBextractor.nLevels");
  int fIniThFAST = Setting::Get<int>("ORBextractor.iniThFAST");
  int fMinThFAST = Setting::Get<int>("ORBextractor.minThFAST");
  orb_extractor_ = ORBextractor::Ptr(new ORBextractor(
      num_orb_bew_features, scale_factor, n_levels, fIniThFAST, fMinThFAST));
}

void LoopClosing::LoadParam()
{
  open_loop_closing_ = Setting::Get<int>("Loop.Closing.Open");
  show_loop_closing_result_ = Setting::Get<int>("Loop.Show.Closing.Result");
  loop_threshold_heigher_ = Setting::Get<float>("Loop.Threshold.Heigher");
  loop_threshold_lower_ = Setting::Get<float>("Loop.Threshold.Lower");
  pyramid_level_num_ = Setting::Get<int>("Pyramid.Level");
  keyframe_database_min_size_ =
      Setting::Get<int>("Loop.Closig.Keyframe.Database.Min.Size");
  LOG(INFO) << "Loop.Closing.Open: " << open_loop_closing_;
  LOG(INFO) << "Loop.Show.Closing.Result: " << show_loop_closing_result_;
  LOG(INFO) << "Loop.Threshold.Heigher: " << loop_threshold_heigher_;
  LOG(INFO) << "Loop.Threshold.Lower: " << loop_threshold_lower_;
  LOG(INFO) << "Pyramid.Level: " << pyramid_level_num_;
  LOG(INFO) << "Loop.Closig.Keyframe.Database.Min.Size: " << keyframe_database_min_size_;
}

} // namespace ssvio