
#include <opencv2/core.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <semantic_perception/semantic_node.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

/**
 * @file semantic_node.cpp
 * @brief Implementation of the SemanticPerception node for 3D semantic
 * segmentation and detection.
 */

namespace SemanticPerception {

namespace {

const char *boolToString(bool value) { return value ? "true" : "false"; }

double stampToSec(const rclcpp::Time &stamp) { return stamp.seconds(); }

double stampDeltaMs(const rclcpp::Time &lhs, const rclcpp::Time &rhs) {
  return std::abs((lhs - rhs).seconds()) * 1000.0;
}

std::string displayFrameId(const std::string &frame_id) {
  return frame_id.empty() ? std::string("<empty>") : frame_id;
}

std::string dominantRejectionReason(
    const std::unordered_map<std::string, size_t> &rejection_counts) {
  std::string dominant_reason;
  size_t dominant_count = 0U;
  for (const auto &entry : rejection_counts) {
    const bool better_count = entry.second > dominant_count;
    const bool tie_breaker =
        entry.second == dominant_count &&
        (dominant_reason.empty() || entry.first < dominant_reason);
    if (better_count || tie_breaker) {
      dominant_reason = entry.first;
      dominant_count = entry.second;
    }
  }
  if (dominant_reason.empty()) {
    return "none_recorded";
  }
  return dominant_reason + " (" + std::to_string(dominant_count) + ")";
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

rmw_qos_reliability_policy_t parseReliability(const std::string &value,
                                              const rclcpp::Logger &logger) {
  const auto lowered = toLower(value);
  if (lowered == "best_effort" || lowered == "best-effort") {
    return RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
  }
  if (lowered == "reliable") {
    return RMW_QOS_POLICY_RELIABILITY_RELIABLE;
  }
  RCLCPP_WARN(logger,
              "Unknown reliability policy '%s', falling back to system default",
              value.c_str());
  return RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT;
}

rmw_qos_durability_policy_t parseDurability(const std::string &value,
                                            const rclcpp::Logger &logger) {
  const auto lowered = toLower(value);
  if (lowered == "transient_local" || lowered == "transient-local") {
    return RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
  }
  if (lowered == "volatile") {
    return RMW_QOS_POLICY_DURABILITY_VOLATILE;
  }
  RCLCPP_WARN(logger,
              "Unknown durability policy '%s', falling back to system default",
              value.c_str());
  return RMW_QOS_POLICY_DURABILITY_SYSTEM_DEFAULT;
}

rmw_qos_history_policy_t parseHistory(const std::string &value,
                                      const rclcpp::Logger &logger) {
  const auto lowered = toLower(value);
  if (lowered == "keep_last" || lowered == "keep-last") {
    return RMW_QOS_POLICY_HISTORY_KEEP_LAST;
  }
  if (lowered == "keep_all" || lowered == "keep-all") {
    return RMW_QOS_POLICY_HISTORY_KEEP_ALL;
  }
  RCLCPP_WARN(logger,
              "Unknown history policy '%s', falling back to system default",
              value.c_str());
  return RMW_QOS_POLICY_HISTORY_SYSTEM_DEFAULT;
}

int clampIntParam(const rclcpp::Logger &logger, const char *name, int value,
                  int minimum) {
  if (value >= minimum) {
    return value;
  }
  RCLCPP_WARN(logger, "Parameter '%s'=%d is invalid; clamping to %d", name,
              value, minimum);
  return minimum;
}

float clampFloatParam(const rclcpp::Logger &logger, const char *name,
                      float value, float minimum, float maximum) {
  if (value >= minimum && value <= maximum) {
    return value;
  }
  const auto clamped = std::clamp(value, minimum, maximum);
  RCLCPP_WARN(logger,
              "Parameter '%s'=%f is out of range [%f, %f]; clamping to %f",
              name, value, minimum, maximum, clamped);
  return clamped;
}

double clampDoubleParam(const rclcpp::Logger &logger, const char *name,
                        double value, double minimum) {
  if (value >= minimum) {
    return value;
  }
  RCLCPP_WARN(logger, "Parameter '%s'=%f is invalid; clamping to %f", name,
              value, minimum);
  return minimum;
}

rmw_qos_profile_t buildRmwQosProfile(int depth, const std::string &history,
                                     const std::string &reliability,
                                     const std::string &durability,
                                     const rclcpp::Logger &logger) {
  const auto valid_depth = static_cast<size_t>(std::max(depth, 1));
  auto profile = rmw_qos_profile_sensor_data;
  profile.history = parseHistory(history, logger);
  profile.depth = valid_depth;
  profile.reliability = parseReliability(reliability, logger);
  profile.durability = parseDurability(durability, logger);
  return profile;
}

rclcpp::QoS buildPublisherQos(int depth, const std::string &history,
                              const std::string &reliability,
                              const std::string &durability,
                              const rclcpp::Logger &logger) {
  const auto valid_depth = std::max(depth, 1);
  rclcpp::QoS qos{rclcpp::KeepLast(valid_depth)};
  const auto history_policy = parseHistory(history, logger);
  const auto reliability_policy = parseReliability(reliability, logger);
  const auto durability_policy = parseDurability(durability, logger);

  if (history_policy == RMW_QOS_POLICY_HISTORY_KEEP_ALL) {
    qos.keep_all();
  } else if (history_policy == RMW_QOS_POLICY_HISTORY_KEEP_LAST) {
    qos.keep_last(valid_depth);
  }

  if (reliability_policy == RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT) {
    qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  } else if (reliability_policy == RMW_QOS_POLICY_RELIABILITY_RELIABLE) {
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  }

  if (durability_policy == RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL) {
    qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
  } else if (durability_policy == RMW_QOS_POLICY_DURABILITY_VOLATILE) {
    qos.durability(rclcpp::DurabilityPolicy::Volatile);
  }

  return qos;
}

} // namespace

/**
 * @class SemanticPerception
 * @brief ROS2 node for performing semantic segmentation and 3D object
 * detection.
 *
 * This node subscribes to RGB and depth images, segments objects, projects
 * masks to 3D, clusters points, and publishes annotated images and 3D
 * detections.
 */
SemanticPerception::SemanticPerception() : Node("semantic_perception_node") {
  // Topic contract:
  //   Inputs  : rgb_topic + depth_topic (synchronized), depth_info_topic
  //             (one-shot intrinsics)
  //   Outputs : /semantic_node/detections, /semantic_node/annotated_image
  // The detection output is consumed downstream by scene_graph_ros.

  // Declare and get parameters
  declare_parameter("model_file", "path/to/model.onnx");
  declare_parameter("class_file", "path/to/classes.yaml");
  declare_parameter("rgb_topic", "/camera/rgb/image_raw");
  declare_parameter("depth_topic", "/camera/depth/image_raw");
  declare_parameter("depth_info_topic", "/camera/depth/camera_info");
  // Detection parameters
  declare_parameter("conf_thresh", 0.5f);
  declare_parameter("iou_thresh", 0.5f);

  // 3D clustering parameters
  declare_parameter("cluster_eps", 0.01f);
  declare_parameter("cluster_min_size", 500.0f);
  declare_parameter("cluster_max_size", 2500000);

  // Message synchronization
  declare_parameter("queue_size", 10);
  declare_parameter("sync_max_interval_sec", 0.5);

  // Frame configuration
  declare_parameter("target_frame", "odom");
  declare_parameter("tf_lookup_timeout_sec", 0.1);
#ifdef USE_CUDA
  const bool use_gpu_default = true;
#else
  const bool use_gpu_default = false;
#endif
  declare_parameter("use_gpu", use_gpu_default);

  // QoS configuration (defaults tuned for typical depth sensors)
  declare_parameter("rgb_qos_history", "keep_last");
  declare_parameter("rgb_qos_reliability", "best_effort");
  declare_parameter("rgb_qos_durability", "volatile");
  declare_parameter("rgb_qos_depth", -1);
  declare_parameter("depth_qos_history", "keep_last");
  declare_parameter("depth_qos_reliability", "best_effort");
  declare_parameter("depth_qos_durability", "volatile");
  declare_parameter("depth_qos_depth", -1);
  declare_parameter("depth_info_qos_history", "keep_last");
  declare_parameter("depth_info_qos_reliability", "reliable");
  declare_parameter("depth_info_qos_durability", "volatile");
  declare_parameter("depth_info_qos_depth", -1);
  declare_parameter("ann_img_qos_history", "keep_last");
  declare_parameter("ann_img_qos_reliability", "reliable");
  declare_parameter("ann_img_qos_durability", "volatile");
  declare_parameter("ann_img_qos_depth", -1);
  declare_parameter("detections_qos_history", "keep_last");
  declare_parameter("detections_qos_reliability", "reliable");
  declare_parameter("detections_qos_durability", "volatile");
  declare_parameter("detections_qos_depth", -1);

  get_parameter("model_file", model_file_);
  get_parameter("class_file", class_file_);
  get_parameter("rgb_topic", rgb_topic_);
  get_parameter("depth_topic", depth_topic_);
  get_parameter("depth_info_topic", depth_info_topic_);
  get_parameter("conf_thresh", conf_thresh_);
  get_parameter("iou_thresh", iou_thresh_);
  get_parameter("cluster_eps", cluster_eps_);
  get_parameter("cluster_min_size", cluster_min_size_);
  get_parameter("cluster_max_size", cluster_max_size_);
  get_parameter("queue_size", queue_size_);
  get_parameter("sync_max_interval_sec", sync_max_interval_sec_);
  get_parameter("target_frame", target_frame_);
  get_parameter("tf_lookup_timeout_sec", tf_lookup_timeout_sec_);
  get_parameter("use_gpu", use_gpu_);

  // Validate critical parameters to avoid runtime instability.
  queue_size_ = clampIntParam(get_logger(), "queue_size", queue_size_, 1);
  cluster_eps_ = clampFloatParam(get_logger(), "cluster_eps", cluster_eps_,
                                 1.0e-6f, std::numeric_limits<float>::max());
  cluster_min_size_ =
      clampFloatParam(get_logger(), "cluster_min_size", cluster_min_size_, 1.0f,
                      std::numeric_limits<float>::max());
  cluster_max_size_ =
      clampIntParam(get_logger(), "cluster_max_size", cluster_max_size_, 1);
  const int min_cluster_size = std::max(1, static_cast<int>(cluster_min_size_));
  if (cluster_max_size_ < min_cluster_size) {
    RCLCPP_WARN(
        get_logger(),
        "Parameter 'cluster_max_size'=%d is smaller than cluster_min_size=%d; "
        "clamping to %d",
        cluster_max_size_, min_cluster_size, min_cluster_size);
    cluster_max_size_ = min_cluster_size;
  }
  sync_max_interval_sec_ = clampDoubleParam(
      get_logger(), "sync_max_interval_sec", sync_max_interval_sec_, 0.0);
  tf_lookup_timeout_sec_ = clampDoubleParam(
      get_logger(), "tf_lookup_timeout_sec", tf_lookup_timeout_sec_, 0.0);

  std::string rgb_qos_history;
  std::string rgb_qos_reliability;
  std::string rgb_qos_durability;
  std::string depth_qos_history;
  std::string depth_qos_reliability;
  std::string depth_qos_durability;
  std::string depth_info_qos_history;
  std::string depth_info_qos_reliability;
  std::string depth_info_qos_durability;
  std::string ann_img_qos_history;
  std::string ann_img_qos_reliability;
  std::string ann_img_qos_durability;
  std::string detections_qos_history;
  std::string detections_qos_reliability;
  std::string detections_qos_durability;

  int rgb_qos_depth_override{0};
  int depth_qos_depth_override{0};
  int depth_info_qos_depth_override{0};
  int ann_img_qos_depth_override{0};
  int detections_qos_depth_override{0};

  get_parameter("rgb_qos_history", rgb_qos_history);
  get_parameter("rgb_qos_reliability", rgb_qos_reliability);
  get_parameter("rgb_qos_durability", rgb_qos_durability);
  get_parameter("rgb_qos_depth", rgb_qos_depth_override);
  get_parameter("depth_qos_history", depth_qos_history);
  get_parameter("depth_qos_reliability", depth_qos_reliability);
  get_parameter("depth_qos_durability", depth_qos_durability);
  get_parameter("depth_qos_depth", depth_qos_depth_override);
  get_parameter("depth_info_qos_history", depth_info_qos_history);
  get_parameter("depth_info_qos_reliability", depth_info_qos_reliability);
  get_parameter("depth_info_qos_durability", depth_info_qos_durability);
  get_parameter("depth_info_qos_depth", depth_info_qos_depth_override);
  get_parameter("ann_img_qos_history", ann_img_qos_history);
  get_parameter("ann_img_qos_reliability", ann_img_qos_reliability);
  get_parameter("ann_img_qos_durability", ann_img_qos_durability);
  get_parameter("ann_img_qos_depth", ann_img_qos_depth_override);
  get_parameter("detections_qos_history", detections_qos_history);
  get_parameter("detections_qos_reliability", detections_qos_reliability);
  get_parameter("detections_qos_durability", detections_qos_durability);
  get_parameter("detections_qos_depth", detections_qos_depth_override);

  const int rgb_qos_depth =
      rgb_qos_depth_override > 0 ? rgb_qos_depth_override : queue_size_;
  const int depth_qos_depth =
      depth_qos_depth_override > 0 ? depth_qos_depth_override : queue_size_;
  const int depth_info_qos_depth = depth_info_qos_depth_override > 0
                                       ? depth_info_qos_depth_override
                                       : queue_size_;
  const int ann_img_qos_depth =
      ann_img_qos_depth_override > 0 ? ann_img_qos_depth_override : queue_size_;
  const int detections_qos_depth = detections_qos_depth_override > 0
                                       ? detections_qos_depth_override
                                       : queue_size_;

  const auto detections_reliability_policy =
      parseReliability(detections_qos_reliability, this->get_logger());
  if (detections_reliability_policy !=
      RMW_QOS_POLICY_RELIABILITY_RELIABLE) {
    RCLCPP_WARN(
        this->get_logger(),
        "detections_qos_reliability is '%s'. scene_graph_ros default "
        "subscriber QoS is RELIABLE; with BEST_EFFORT publisher QoS, detections "
        "may not be delivered.",
        detections_qos_reliability.c_str());
  }

  const auto rgb_qos_profile =
      buildRmwQosProfile(rgb_qos_depth, rgb_qos_history, rgb_qos_reliability,
                         rgb_qos_durability, this->get_logger());
  const auto depth_qos_profile = buildRmwQosProfile(
      depth_qos_depth, depth_qos_history, depth_qos_reliability,
      depth_qos_durability, this->get_logger());
  const auto depth_info_qos_profile = buildRmwQosProfile(
      depth_info_qos_depth, depth_info_qos_history, depth_info_qos_reliability,
      depth_info_qos_durability, this->get_logger());

  const auto ann_img_qos =
      buildPublisherQos(ann_img_qos_depth, ann_img_qos_history,
                        ann_img_qos_reliability, ann_img_qos_durability,
                        this->get_logger());
  const auto detections_qos =
      buildPublisherQos(detections_qos_depth, detections_qos_history,
                        detections_qos_reliability, detections_qos_durability,
                        this->get_logger());

  // Initialize TF2 with extended cache duration for handling timing issues
  // Default is 10 seconds, increasing to 30 seconds to accommodate delays
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(
      this->get_clock(), tf2::Duration(std::chrono::seconds(30)));
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Load segmentation model and class file
  loadModel(model_file_, class_file_);

  // Set up subscribers
  rgb_sub_.subscribe(this, rgb_topic_, rgb_qos_profile);
  depth_sub_.subscribe(this, depth_topic_, depth_qos_profile);

  const auto depth_info_qos = rclcpp::QoS(
      rclcpp::QoSInitialization::from_rmw(depth_info_qos_profile),
      depth_info_qos_profile);
  depth_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      depth_info_topic_, depth_info_qos,
      std::bind(&SemanticPerception::depthInfoCallback, this,
                std::placeholders::_1));

  // Set up synchronizer for RGB and depth.
  sync_ = std::make_shared<Sync>(SyncPolicy(queue_size_), rgb_sub_, depth_sub_);
  sync_->setMaxIntervalDuration(
      rclcpp::Duration::from_seconds(sync_max_interval_sec_));
  sync_->registerCallback(std::bind(&SemanticPerception::syncCallback, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2));

  // Set up publishers
  ann_img_pub_ =
      create_publisher<sensor_msgs::msg::Image>(ann_img_topic_, ann_img_qos);
  det_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
      det_topic_, detections_qos);
}

/**
 * @brief Destructor for SemanticPerception node.
 */
SemanticPerception::~SemanticPerception() = default;

/**
 * @brief Loads the segmentation model and class file.
 * @param model_ Path to the model file.
 * @param class_file Path to the class file.
 */
void SemanticPerception::loadModel(const std::string &model_,
                                   const std::string &class_file) {
  std::filesystem::path pkg_share =
      ament_index_cpp::get_package_share_directory("semantic_perception");
  if (pkg_share.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Package share directory not found.");
    return;
  }

  std::filesystem::path model_path = pkg_share / "models" / model_;
  std::filesystem::path class_path = pkg_share / "models" / class_file;

  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(class_path)) {
    RCLCPP_FATAL(get_logger(), "Model or class file not found, shutting down node");
    throw std::runtime_error("Semantic model files missing");
  }

  const bool gpu_requested = use_gpu_;
#ifdef USE_CUDA
  const bool gpu_build_enabled = true;
#else
  const bool gpu_build_enabled = false;
#endif

  const auto providers = Ort::GetAvailableProviders();
  const bool cuda_provider_available =
      std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
      providers.end();

  std::ostringstream providers_stream;
  for (size_t i = 0; i < providers.size(); ++i) {
    if (i > 0) {
      providers_stream << ", ";
    }
    providers_stream << providers[i];
  }
  const auto providers_str = providers_stream.str();

  if (gpu_requested && !gpu_build_enabled) {
    RCLCPP_WARN(
        this->get_logger(),
        "use_gpu=true but semantic_perception was built without "
        "-DONNXRUNTIME_USE_GPU=ON; forcing CPU inference.");
  }
  if (gpu_requested && gpu_build_enabled && !cuda_provider_available) {
    RCLCPP_WARN(
        this->get_logger(),
        "use_gpu=true and build supports GPU, but CUDAExecutionProvider is not "
        "available in current ONNX Runtime. This usually means ONNXRUNTIME_DIR "
        "points to a CPU runtime or CUDA dependencies are missing. "
        "Available providers: [%s]",
        providers_str.c_str());
  }

  const bool gpu_enabled = gpu_requested && gpu_build_enabled;
  RCLCPP_INFO(
      this->get_logger(),
      "Loading model: %s | use_gpu(requested)=%s | build_gpu=%s | "
      "cuda_provider_available=%s",
      model_path.string().c_str(), gpu_requested ? "true" : "false",
      gpu_build_enabled ? "true" : "false",
      cuda_provider_available ? "true" : "false");

  segmentor_ = std::make_shared<yolos::seg::YOLOSegDetector>(
      model_path.string(), class_path.string(), gpu_enabled);
  if (!segmentor_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s",
                 model_path.string().c_str());
    return;
  }

  const auto actual_device = segmentor_->getDevice();
  RCLCPP_INFO(this->get_logger(),
              "semantic_perception inference device selected by ORT: %s",
              actual_device.c_str());
  if (gpu_requested && actual_device != "gpu") {
    RCLCPP_WARN(
        this->get_logger(),
        "GPU was requested but inference is running on CPU. "
        "Available providers: [%s]",
        providers_str.c_str());
  }
}

void SemanticPerception::depthInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg) {
  if (has_camera_intrinsics_) {
    return;
  }

  const double fx = depth_info_msg->k[0];
  const double fy = depth_info_msg->k[4];
  const double cx = depth_info_msg->k[2];
  const double cy = depth_info_msg->k[5];

  if (fx <= 0.0 || fy <= 0.0 || !std::isfinite(fx) || !std::isfinite(fy) ||
      !std::isfinite(cx) || !std::isfinite(cy)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Received invalid camera intrinsics from '%s': "
                         "fx=%f, fy=%f, cx=%f, cy=%f",
                         depth_info_topic_.c_str(), fx, fy, cx, cy);
    return;
  }

  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  depth_info_frame_id_ = depth_info_msg->header.frame_id;
  has_camera_intrinsics_ = true;

  RCLCPP_INFO(get_logger(),
              "Cached depth intrinsics from '%s': fx=%.3f fy=%.3f cx=%.3f "
              "cy=%.3f frame='%s'",
              depth_info_topic_.c_str(), fx_, fy_, cx_, cy_,
              depth_info_frame_id_.c_str());

  // One-shot subscription: intrinsics are static, so stop receiving updates.
  depth_info_sub_.reset();
  RCLCPP_INFO(get_logger(), "Unsubscribed from depth camera info topic '%s'",
              depth_info_topic_.c_str());
}

/**
 * @brief Callback for synchronized RGB and depth messages.
 *
 * Performs segmentation, projects masks to 3D, clusters points, and publishes
 * results.
 *
 * @param rgb_msg Shared pointer to RGB image message.
 * @param depth_msg Shared pointer to depth image message.
 */
void SemanticPerception::syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg) {
  struct CallbackDebugStats {
    size_t detector_results{0U};
    bool tf_needed{false};
    bool tf_success{false};
    std::string tf_status{"not_required"};
    size_t valid_3d_points{0U};
    size_t valid_clusters{0U};
    size_t detections_published{0U};
    std::string callback_status{"started"};
    std::unordered_map<std::string, size_t> rejection_counts;
  } debug_stats;

  auto record_rejection = [&](const std::string &reason) {
    ++debug_stats.rejection_counts[reason];
  };

  auto log_summary = [&](const std::string &status) {
    debug_stats.callback_status = status;
    const std::string dominant_reason =
        debug_stats.detections_published == 0U
            ? dominantRejectionReason(debug_stats.rejection_counts)
            : "n/a";
    RCLCPP_INFO(
        get_logger(),
        "[semantic_debug][summary] status=%s detector_results=%zu tf=%s "
        "valid_3d_points=%zu valid_clusters=%zu detections_published=%zu "
        "dominant_rejection_reason=%s",
        debug_stats.callback_status.c_str(), debug_stats.detector_results,
        debug_stats.tf_status.c_str(), debug_stats.valid_3d_points,
        debug_stats.valid_clusters, debug_stats.detections_published,
        dominant_reason.c_str());
  };

  try {
    vision_msgs::msg::Detection3DArray detections_msg;
    const bool publish_annotated =
        ann_img_pub_->get_subscription_count() > 0U ||
        ann_img_pub_->get_intra_process_subscription_count() > 0U;

    const auto rgb_stamp = rclcpp::Time(rgb_msg->header.stamp);
    const auto depth_stamp = rclcpp::Time(depth_msg->header.stamp);
    const std::string rgb_frame = displayFrameId(rgb_msg->header.frame_id);
    const std::string depth_frame = displayFrameId(depth_msg->header.frame_id);
    const std::string depth_info_frame_log = displayFrameId(depth_info_frame_id_);

    RCLCPP_DEBUG(
        get_logger(),
        "[semantic_debug][sync_input] rgb(frame='%s', stamp=%.6f, encoding='%s', "
        "%ux%u) depth(frame='%s', stamp=%.6f, encoding='%s', %ux%u) "
        "camera_info(frame='%s', cached=%s) delta_ms(rgb-depth)=%.3f "
        "publish_annotated=%s ann_subscribers=%zu det_subscribers=%zu",
        rgb_frame.c_str(), stampToSec(rgb_stamp), rgb_msg->encoding.c_str(),
        rgb_msg->width, rgb_msg->height, depth_frame.c_str(),
        stampToSec(depth_stamp), depth_msg->encoding.c_str(), depth_msg->width,
        depth_msg->height, depth_info_frame_log.c_str(),
        boolToString(has_camera_intrinsics_), stampDeltaMs(rgb_stamp, depth_stamp),
        boolToString(publish_annotated), ann_img_pub_->get_subscription_count(),
        det_pub_->get_subscription_count());

    if (!has_camera_intrinsics_) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Waiting for first depth camera info on '%s'; skipping RGB/depth "
          "pair",
          depth_info_topic_.c_str());
      record_rejection("camera_intrinsics_not_cached");
      detections_msg.header.frame_id = target_frame_;
      detections_msg.header.stamp = depth_msg->header.stamp;
      det_pub_->publish(detections_msg);
      log_summary("camera_intrinsics_not_cached");
      return;
    }

    // Convert ROS images to OpenCV.
    const auto rgb_cv =
        cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
    const cv::Mat &input_image = rgb_cv->image;
    if (input_image.empty()) {
      RCLCPP_WARN(get_logger(),
                  "[semantic_debug][input] RGB image is empty, skipping callback");
      record_rejection("empty_rgb_image");
      log_summary("empty_rgb_image");
      return;
    }

    cv_bridge::CvImageConstPtr depth_cv;
    bool depth_is_float = false;
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      depth_cv = cv_bridge::toCvShare(
          depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
      depth_is_float = true;
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      depth_cv = cv_bridge::toCvShare(
          depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } else {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                            "Unsupported depth encoding: %s",
                            depth_msg->encoding.c_str());
      record_rejection("unsupported_depth_encoding");
      log_summary("unsupported_depth_encoding");
      return;
    }

    const cv::Mat &depth_image = depth_cv->image;
    if (depth_image.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Received empty depth image");
      record_rejection("empty_depth_image");
      log_summary("empty_depth_image");
      return;
    }

    const std::string source_frame = depth_msg->header.frame_id;
    const std::string &depth_info_frame = depth_info_frame_id_;
    if (!source_frame.empty() && !depth_info_frame.empty() &&
        source_frame != depth_info_frame) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Depth frame mismatch: depth image frame '%s' != depth camera_info "
          "frame '%s'. Using depth image frame for 3D detections.",
          source_frame.c_str(), depth_info_frame.c_str());
    }

    // Segment the new image.
    auto results = segmentor_->segment(input_image, conf_thresh_, iou_thresh_);
    debug_stats.detector_results = results.size();
    RCLCPP_DEBUG(get_logger(),
                 "[semantic_debug][detector] segmentor produced %zu results",
                 results.size());

    const auto &class_names = segmentor_->getClassNames();
    for (size_t result_index = 0; result_index < results.size(); ++result_index) {
      const auto &result = results[result_index];
      const int class_id = result.classId;
      std::string class_name_for_log = "<invalid_class_id>";
      if (class_id >= 0 && static_cast<size_t>(class_id) < class_names.size()) {
        const auto &raw_name = class_names[static_cast<size_t>(class_id)];
        class_name_for_log =
            raw_name.empty() ? ("class_" + std::to_string(class_id)) : raw_name;
      }

      const cv::Mat &mask = result.mask;
      const bool mask_empty = mask.empty();
      const int mask_type = mask_empty ? -1 : mask.type();
      const int mask_cols = mask_empty ? 0 : mask.cols;
      const int mask_rows = mask_empty ? 0 : mask.rows;
      const bool mask_valid = !mask_empty && mask_type == CV_8UC1;
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][detector_result] idx=%zu class_id=%d label='%s' "
          "conf=%.3f box=[x=%d,y=%d,w=%d,h=%d] mask_empty=%s mask_type=%d "
          "mask_size=%dx%d mask_valid=%s",
          result_index, class_id, class_name_for_log.c_str(), result.conf,
          result.box.x, result.box.y, result.box.width, result.box.height,
          boolToString(mask_empty), mask_type, mask_cols, mask_rows,
          boolToString(mask_valid));
    }

    // Publish annotated output right after segmentation, before 3D processing.
    if (publish_annotated) {
      if (results.empty()) {
        ann_img_pub_->publish(*rgb_msg);
        RCLCPP_DEBUG(
            get_logger(),
            "[semantic_debug][publish] published pass-through annotated image "
            "frame='%s' stamp=%.6f",
            rgb_frame.c_str(), stampToSec(rgb_stamp));
      } else {
        cv::Mat ann_image = input_image.clone();
        segmentor_->drawSegmentations(ann_image, results);
        ann_img_pub_->publish(
            *cv_bridge::CvImage(rgb_msg->header,
                                sensor_msgs::image_encodings::BGR8, ann_image)
                 .toImageMsg());
        RCLCPP_DEBUG(
            get_logger(),
            "[semantic_debug][publish] published annotated image frame='%s' "
            "stamp=%.6f",
            rgb_frame.c_str(), stampToSec(rgb_stamp));
      }
    } else {
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][publish] skipped annotated image publish "
          "(no subscribers)");
    }

    // If no results, publish empty detections and pass-through image.
    if (results.empty()) {
      record_rejection("no_detector_results");
      detections_msg.header.frame_id = target_frame_;
      detections_msg.header.stamp = depth_msg->header.stamp;
      debug_stats.detections_published = 0U;
      RCLCPP_DEBUG(get_logger(),
                   "[semantic_debug][publish] publishing empty detections: "
                   "frame='%s' stamp=%.6f",
                   target_frame_.c_str(), stampToSec(depth_stamp));
      det_pub_->publish(detections_msg);
      log_summary("no_detector_results");
      return;
    }

    if (source_frame.empty()) {
      debug_stats.tf_needed = true;
      debug_stats.tf_success = false;
      debug_stats.tf_status = "failure";
      record_rejection("empty_source_frame");
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Dropping detections: depth image frame_id is empty, cannot transform "
          "to target frame '%s'",
          target_frame_.c_str());
      detections_msg.header.frame_id = target_frame_;
      detections_msg.header.stamp = depth_msg->header.stamp;
      debug_stats.detections_published = 0U;
      RCLCPP_DEBUG(get_logger(),
                   "[semantic_debug][publish] publishing empty detections due to "
                   "empty source frame: target_frame='%s' stamp=%.6f",
                   target_frame_.c_str(), stampToSec(depth_stamp));
      det_pub_->publish(detections_msg);
      log_summary("empty_source_frame");
      return;
    }

    const bool needs_transform = source_frame != target_frame_;
    debug_stats.tf_needed = needs_transform;
    debug_stats.tf_success = !needs_transform;
    debug_stats.tf_status = needs_transform ? "pending" : "not_required";
    bool have_transform = !needs_transform;
    geometry_msgs::msg::TransformStamped camera_to_target_tf;
    if (needs_transform) {
      const auto timeout = tf2::durationFromSec(tf_lookup_timeout_sec_);
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][tf_lookup] requesting exact transform source='%s' "
          "target='%s' lookup_stamp=%.6f timeout_sec=%.3f",
          source_frame.c_str(), target_frame_.c_str(), stampToSec(depth_stamp),
          tf_lookup_timeout_sec_);
      try {
        camera_to_target_tf = tf_buffer_->lookupTransform(
            target_frame_, source_frame,
            tf2_ros::fromMsg(depth_msg->header.stamp), timeout);
        have_transform = true;
        debug_stats.tf_success = true;
        debug_stats.tf_status = "success";
        RCLCPP_DEBUG(
            get_logger(),
            "[semantic_debug][tf_lookup] success source='%s' target='%s' "
            "lookup_stamp=%.6f tf_stamp=%.6f tf_header_frame='%s' "
            "tf_child_frame='%s'",
            source_frame.c_str(), target_frame_.c_str(), stampToSec(depth_stamp),
            stampToSec(rclcpp::Time(camera_to_target_tf.header.stamp)),
            camera_to_target_tf.header.frame_id.c_str(),
            camera_to_target_tf.child_frame_id.c_str());
      } catch (const tf2::TransformException &exact_ex) {
        debug_stats.tf_success = false;
        debug_stats.tf_status = "failure";
        record_rejection("tf_exact_lookup_failed");
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Dropping frame: exact-time TF lookup from '%s' to '%s' at stamp "
            "%.6f with timeout %.3f sec failed (%s)",
            source_frame.c_str(), target_frame_.c_str(), stampToSec(depth_stamp),
            tf_lookup_timeout_sec_, exact_ex.what());
      }
    } else {
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][tf_lookup] skipped lookup because source and target "
          "frames match ('%s')",
          source_frame.c_str());
    }

    if (needs_transform && !have_transform) {
      detections_msg.header.frame_id = target_frame_;
      detections_msg.header.stamp = depth_msg->header.stamp;
      debug_stats.detections_published = 0U;
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][publish] publishing empty detections due to TF "
          "lookup failure: target_frame='%s' stamp=%.6f",
          target_frame_.c_str(), stampToSec(depth_stamp));
      det_pub_->publish(detections_msg);
      log_summary("tf_lookup_failed");
      return;
    }

    // Process results and publish annotated image and detections.
    RCLCPP_DEBUG(get_logger(),
                 "[semantic_debug][pipeline] processing %zu detector results",
                 results.size());
    detections_msg.detections.reserve(results.size());
    const int depth_cols = depth_image.cols;
    const int depth_rows = depth_image.rows;
    const int min_cluster_size = std::max(1, static_cast<int>(cluster_min_size_));

    std::vector<int> depth_x_lut;
    std::vector<int> depth_y_lut;

    for (size_t result_index = 0; result_index < results.size(); ++result_index) {
      const auto &result = results[result_index];
      const int class_id = result.classId;
      if (class_id < 0 || static_cast<size_t>(class_id) >= class_names.size()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Skipping detection with invalid class id %d",
                             class_id);
        record_rejection("invalid_class_id");
        continue;
      }

      const std::string &raw_class_name =
          class_names[static_cast<size_t>(class_id)];
      const std::string class_name = !raw_class_name.empty()
                                         ? raw_class_name
                                         : ("class_" + std::to_string(class_id));
      if (raw_class_name.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Class name for class id %d is empty; using '%s'",
                             class_id, class_name.c_str());
      }
      const float confidence = result.conf;
      const cv::Mat &mask = result.mask;
      if (mask.empty()) {
        RCLCPP_DEBUG(get_logger(),
                     "Skipping detection '%s': segmentation mask is empty",
                     class_name.c_str());
        record_rejection("empty_mask");
        continue;
      }
      if (mask.type() != CV_8UC1) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Skipping detection '%s': unexpected mask type %d",
                             class_name.c_str(), mask.type());
        record_rejection("invalid_mask_type");
        continue;
      }
      if (mask.cols <= 0 || mask.rows <= 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Skipping detection '%s': invalid mask dimensions "
                             "%dx%d",
                             class_name.c_str(), mask.cols, mask.rows);
        record_rejection("invalid_mask_dimensions");
        continue;
      }

      const int x_offset = result.box.x;
      const int y_offset = result.box.y;
      const int width = result.box.width;
      const int height = result.box.height;

      // Calculate scaling factors for resolution mismatch.
      const float scale_x =
          static_cast<float>(depth_cols) / static_cast<float>(mask.cols);
      const float scale_y =
          static_cast<float>(depth_rows) / static_cast<float>(mask.rows);
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][registration] idx=%zu class='%s' rgb_size=%dx%d "
          "depth_size=%dx%d mask_size=%dx%d scale=(%.6f,%.6f) "
          "box=[x=%d,y=%d,w=%d,h=%d]",
          result_index, class_name.c_str(), input_image.cols, input_image.rows,
          depth_cols, depth_rows, mask.cols, mask.rows, scale_x, scale_y,
          x_offset, y_offset, width, height);

      const int x_min =
          std::max(0, std::min(mask.cols, static_cast<int>(x_offset)));
      const int y_min =
          std::max(0, std::min(mask.rows, static_cast<int>(y_offset)));
      const int x_max =
          std::min(mask.cols, std::max(x_min, x_min + static_cast<int>(width)));
      const int y_max =
          std::min(mask.rows, std::max(y_min, y_min + static_cast<int>(height)));
      if (x_min >= x_max || y_min >= y_max) {
        RCLCPP_DEBUG(get_logger(),
                     "Skipping detection '%s': invalid ROI bounds (%d,%d,%d,%d)",
                     class_name.c_str(), x_min, y_min, x_max, y_max);
        record_rejection("invalid_roi_bounds");
        continue;
      }
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][registration] idx=%zu class='%s' roi=[x_min=%d, "
          "y_min=%d, x_max=%d, y_max=%d]",
          result_index, class_name.c_str(), x_min, y_min, x_max, y_max);

      // Precompute x/y mappings from mask to depth coordinates once per ROI.
      if (static_cast<int>(depth_x_lut.size()) < mask.cols) {
        depth_x_lut.resize(mask.cols);
      }
      for (int x = x_min; x < x_max; ++x) {
        const int mapped_x = static_cast<int>(static_cast<float>(x) * scale_x);
        depth_x_lut[x] = std::clamp(mapped_x, 0, depth_cols - 1);
      }
      const int roi_height = y_max - y_min;
      if (static_cast<int>(depth_y_lut.size()) < roi_height) {
        depth_y_lut.resize(roi_height);
      }
      for (int i = 0; i < roi_height; ++i) {
        const int y = y_min + i;
        const int mapped_y = static_cast<int>(static_cast<float>(y) * scale_y);
        depth_y_lut[i] = std::clamp(mapped_y, 0, depth_rows - 1);
      }

      auto cloud_seg =
          pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
      cloud_seg->points.reserve(
          static_cast<size_t>(x_max - x_min) * static_cast<size_t>(y_max - y_min));
      size_t masked_pixels = 0U;
      size_t invalid_depth_values = 0U;
      size_t invalid_projected_points = 0U;

      if (depth_is_float) {
        for (int i = 0; i < roi_height; ++i) {
          const int y = y_min + i;
          const uchar *mask_ptr = mask.ptr<uchar>(y);
          const int depth_y = depth_y_lut[i];
          const float *depth_ptr = depth_image.ptr<float>(depth_y);
          for (int x = x_min; x < x_max; ++x) {
            if (mask_ptr[x] == 0U) {
              continue;
            }
            ++masked_pixels;
            const int depth_x = depth_x_lut[x];
            const float z = depth_ptr[depth_x];
            if (z <= 0.001f || !std::isfinite(z)) {
              ++invalid_depth_values;
              continue;
            }

            pcl::PointXYZ pt;
            pt.x = (static_cast<float>(depth_x) - static_cast<float>(cx_)) * z /
                   static_cast<float>(fx_);
            pt.y = (static_cast<float>(depth_y) - static_cast<float>(cy_)) * z /
                   static_cast<float>(fy_);
            pt.z = z;
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
                !std::isfinite(pt.z)) {
              ++invalid_projected_points;
              continue;
            }
            cloud_seg->points.emplace_back(pt);
          }
        }
      } else {
        for (int i = 0; i < roi_height; ++i) {
          const int y = y_min + i;
          const uchar *mask_ptr = mask.ptr<uchar>(y);
          const int depth_y = depth_y_lut[i];
          const uint16_t *depth_ptr = depth_image.ptr<uint16_t>(depth_y);
          for (int x = x_min; x < x_max; ++x) {
            if (mask_ptr[x] == 0U) {
              continue;
            }
            ++masked_pixels;
            const int depth_x = depth_x_lut[x];
            const float z = static_cast<float>(depth_ptr[depth_x]) * 0.001f;
            if (z <= 0.001f) {
              ++invalid_depth_values;
              continue;
            }

            pcl::PointXYZ pt;
            pt.x = (static_cast<float>(depth_x) - static_cast<float>(cx_)) * z /
                   static_cast<float>(fx_);
            pt.y = (static_cast<float>(depth_y) - static_cast<float>(cy_)) * z /
                   static_cast<float>(fy_);
            pt.z = z;
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
                !std::isfinite(pt.z)) {
              ++invalid_projected_points;
              continue;
            }
            cloud_seg->points.emplace_back(pt);
          }
        }
      }

      const size_t valid_depth_points = cloud_seg->points.size();
      debug_stats.valid_3d_points += valid_depth_points;
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][registration] idx=%zu class='%s' masked_pixels=%zu "
          "valid_depth_points=%zu invalid_depth_values=%zu "
          "invalid_projected_points=%zu",
          result_index, class_name.c_str(), masked_pixels, valid_depth_points,
          invalid_depth_values, invalid_projected_points);

      if (cloud_seg->points.empty()) {
        RCLCPP_DEBUG(get_logger(),
                     "Skipping detection '%s': point cloud is empty",
                     class_name.c_str());
        record_rejection("no_valid_3d_points");
        continue;
      }

      cloud_seg->width = static_cast<uint32_t>(cloud_seg->points.size());
      cloud_seg->height = 1;
      cloud_seg->is_dense = false;

      // Cluster points to remove noise and isolate objects.
      auto tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(
          new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(cloud_seg);

      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(cluster_eps_); // Cluster tolerance (meters)
      ec.setMinClusterSize(min_cluster_size);
      ec.setMaxClusterSize(cluster_max_size_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_seg);

      std::vector<pcl::PointIndices> clusters;
      ec.extract(clusters);

      size_t largest_cluster_points = 0U;
      for (const auto &cluster : clusters) {
        largest_cluster_points =
            std::max(largest_cluster_points, cluster.indices.size());
      }
      debug_stats.valid_clusters += clusters.size();
      RCLCPP_DEBUG(
          get_logger(),
          "[semantic_debug][clustering] idx=%zu class='%s' input_points=%zu "
          "clusters_found=%zu largest_cluster_points=%zu eps=%.3f "
          "min_size=%d max_size=%d",
          result_index, class_name.c_str(), cloud_seg->points.size(),
          clusters.size(), largest_cluster_points, cluster_eps_, min_cluster_size,
          cluster_max_size_);

      if (clusters.empty()) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Skipping detection '%s': no valid clusters found (cloud had %zu "
            "points, needed >=%d)",
            class_name.c_str(), cloud_seg->points.size(), min_cluster_size);
        record_rejection("no_valid_clusters");
        continue;
      }

      const auto best_cluster_it =
          std::max_element(clusters.begin(), clusters.end(),
                           [](const pcl::PointIndices &lhs,
                              const pcl::PointIndices &rhs) {
                             return lhs.indices.size() < rhs.indices.size();
                           });
      if (best_cluster_it == clusters.end() || best_cluster_it->indices.empty()) {
        RCLCPP_DEBUG(get_logger(),
                     "Skipping detection '%s': largest cluster is empty",
                     class_name.c_str());
        record_rejection("empty_largest_cluster");
        continue;
      }

      pcl::PointXYZ aabb_min;
      pcl::PointXYZ aabb_max;
      aabb_min.x = aabb_min.y = aabb_min.z = std::numeric_limits<float>::max();
      aabb_max.x = aabb_max.y = aabb_max.z =
          std::numeric_limits<float>::lowest();
      bool have_valid_cluster_point = false;
      for (const int index : best_cluster_it->indices) {
        if (index < 0 || static_cast<size_t>(index) >= cloud_seg->points.size()) {
          continue;
        }
        const auto &pt = cloud_seg->points[static_cast<size_t>(index)];
        aabb_min.x = std::min(aabb_min.x, pt.x);
        aabb_min.y = std::min(aabb_min.y, pt.y);
        aabb_min.z = std::min(aabb_min.z, pt.z);
        aabb_max.x = std::max(aabb_max.x, pt.x);
        aabb_max.y = std::max(aabb_max.y, pt.y);
        aabb_max.z = std::max(aabb_max.z, pt.z);
        have_valid_cluster_point = true;
      }
      if (!have_valid_cluster_point) {
        RCLCPP_DEBUG(get_logger(),
                     "Skipping detection '%s': largest cluster had no valid points",
                     class_name.c_str());
        record_rejection("largest_cluster_no_valid_points");
        continue;
      }

      // Fill detection message.
      auto detection_msg = vision_msgs::msg::Detection3D();
      detection_msg.header = depth_msg->header;
      detection_msg.header.frame_id = source_frame;
      detection_msg.id = class_name;
      detection_msg.bbox.center.position.x = (aabb_min.x + aabb_max.x) * 0.5f;
      detection_msg.bbox.center.position.y = (aabb_min.y + aabb_max.y) * 0.5f;
      detection_msg.bbox.center.position.z = (aabb_min.z + aabb_max.z) * 0.5f;
      detection_msg.bbox.center.orientation.x = 0.0;
      detection_msg.bbox.center.orientation.y = 0.0;
      detection_msg.bbox.center.orientation.z = 0.0;
      detection_msg.bbox.center.orientation.w = 1.0;
      detection_msg.bbox.size.x = aabb_max.x - aabb_min.x;
      detection_msg.bbox.size.y = aabb_max.y - aabb_min.y;
      detection_msg.bbox.size.z = aabb_max.z - aabb_min.z;

      // Add hypothesis with pose.
      vision_msgs::msg::ObjectHypothesisWithPose hypo;
      hypo.hypothesis.class_id = class_name;
      hypo.hypothesis.score = static_cast<double>(confidence);
      hypo.pose.pose.position = detection_msg.bbox.center.position;
      hypo.pose.pose.orientation = detection_msg.bbox.center.orientation;
      detection_msg.results.emplace_back(std::move(hypo));

      if (needs_transform) {
        if (!have_transform) {
          record_rejection("transform_unavailable_for_detection");
          continue;
        }
        geometry_msgs::msg::PoseStamped pose_in;
        geometry_msgs::msg::PoseStamped pose_out;
        pose_in.header = detection_msg.header;
        pose_in.pose = detection_msg.bbox.center;
        RCLCPP_DEBUG(
            get_logger(),
            "[semantic_debug][tf_apply] class='%s' source_frame='%s' "
            "target_frame='%s' pose_stamp=%.6f tf_stamp=%.6f",
            class_name.c_str(), pose_in.header.frame_id.c_str(),
            target_frame_.c_str(), stampToSec(rclcpp::Time(pose_in.header.stamp)),
            stampToSec(rclcpp::Time(camera_to_target_tf.header.stamp)));
        tf2::doTransform(pose_in, pose_out, camera_to_target_tf);

        detection_msg.header.frame_id = target_frame_;
        detection_msg.header.stamp = pose_out.header.stamp;
        detection_msg.bbox.center = pose_out.pose;
        if (!detection_msg.results.empty()) {
          detection_msg.results[0].pose.pose = pose_out.pose;
        }
        RCLCPP_DEBUG(
            get_logger(),
            "[semantic_debug][tf_apply] class='%s' transformed center=[%.3f, "
            "%.3f, %.3f] frame='%s' stamp=%.6f",
            class_name.c_str(), detection_msg.bbox.center.position.x,
            detection_msg.bbox.center.position.y,
            detection_msg.bbox.center.position.z,
            detection_msg.header.frame_id.c_str(),
            stampToSec(rclcpp::Time(detection_msg.header.stamp)));
      }

      RCLCPP_DEBUG(get_logger(),
                   "Detection: class='%s' (id=%d), conf=%.2f, frame='%s', "
                   "center=[%.2f, %.2f, %.2f], size=[%.2f, %.2f, %.2f]",
                   class_name.c_str(), class_id, confidence,
                   detection_msg.header.frame_id.c_str(),
                   detection_msg.bbox.center.position.x,
                   detection_msg.bbox.center.position.y,
                   detection_msg.bbox.center.position.z,
                   detection_msg.bbox.size.x, detection_msg.bbox.size.y,
                   detection_msg.bbox.size.z);

      detections_msg.detections.emplace_back(std::move(detection_msg));
    }

    if (detections_msg.detections.empty()) {
      record_rejection("no_detection_passed_filters");
    }

    detections_msg.header.frame_id = target_frame_;
    detections_msg.header.stamp = depth_msg->header.stamp;
    debug_stats.detections_published = detections_msg.detections.size();
    RCLCPP_DEBUG(get_logger(),
                 "[semantic_debug][publish] publishing %zu detections "
                 "frame='%s' stamp=%.6f",
                 detections_msg.detections.size(), target_frame_.c_str(),
                 stampToSec(depth_stamp));
    det_pub_->publish(detections_msg);

    log_summary("completed");
  } catch (const std::exception &e) {
    record_rejection("exception");
    RCLCPP_ERROR(get_logger(), "Error in image callback: %s", e.what());
    log_summary("exception");
  }
}

} // namespace SemanticPerception

/**
 * @brief Main entry point for the semantic perception node.
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return int Exit code.
 */
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<SemanticPerception::SemanticPerception>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    std::cerr << "::Exception::" << e.what();
    rclcpp::shutdown();
    return 1;
  }
  return 0;
}
