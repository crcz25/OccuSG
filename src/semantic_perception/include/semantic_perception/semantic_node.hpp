#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>

#include <yolos/tasks/segmentation.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <string>
#include <sys/types.h>
#include <vector>
#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace SemanticPerception {

class SemanticPerception : public rclcpp::Node {
public:
  SemanticPerception();

  // Destructor
  ~SemanticPerception() override;

private:
  void depthInfoCallback(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg);
  void syncCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg);
  void loadModel(const std::string &model_, const std::string &class_file);

  // Parameter cache.
  std::string model_file_, class_file_, rgb_topic_, depth_topic_,
      depth_info_topic_, target_frame_;
  float conf_thresh_, iou_thresh_, cluster_eps_, cluster_min_size_;
  bool use_gpu_{false};

  // Message synchronization parameters
  int queue_size_;
  double sync_max_interval_sec_;
  double tf_lookup_timeout_sec_;

  // TF2 for frame transformations
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // subscriber
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      depth_info_sub_;

  // synchronizer
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  std::shared_ptr<Sync> sync_;

  // publisher
  std::string ann_img_topic_ = "/semantic_node/annotated_image",
              det_topic_ = "/semantic_node/detections";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ann_img_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;

  // storage
  double fx_, fy_, cx_, cy_;
  bool has_camera_intrinsics_{false};
  std::string depth_info_frame_id_;
  int cluster_max_size_{2500000};

  std::shared_ptr<yolos::seg::YOLOSegDetector> segmentor_;
};

} // namespace SemanticPerception
