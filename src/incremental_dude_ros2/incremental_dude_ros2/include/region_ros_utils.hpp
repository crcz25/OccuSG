#ifndef REGION_ROS_UTILS_HPP
#define REGION_ROS_UTILS_HPP

#include "region_geometry_utils.hpp"
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>

/**
 * @brief ROS message conversion utilities
 */
class RegionROSConverter {
public:
  /**
   * @brief Convert OpenCV contour (pixel coords) to ROS Polygon message
   */
  static geometry_msgs::msg::Polygon
  contourToPolygon(const std::vector<cv::Point> &contour,
                   const CoordinateConverter &converter) {
    geometry_msgs::msg::Polygon polygon;
    polygon.points.reserve(contour.size());

    for (const auto &pt : contour) {
      geometry_msgs::msg::Point32 point;
      double map_x, map_y;
      converter.pixelToMap(pt.x, pt.y, map_x, map_y);
      point.x = map_x;
      point.y = map_y;
      point.z = 0.0;
      polygon.points.push_back(point);
    }

    return polygon;
  }

  /**
   * @brief Convert OpenCV point (pixel coords) to ROS Pose2D message
   */
  static geometry_msgs::msg::Pose2D
  pointToPose2D(const cv::Point &point, const CoordinateConverter &converter) {
    geometry_msgs::msg::Pose2D pose;
    double map_x, map_y;
    converter.pixelToMap(point.x, point.y, map_x, map_y);
    pose.x = map_x;
    pose.y = map_y;
    pose.theta = 0.0;
    return pose;
  }
};

#endif // REGION_ROS_UTILS_HPP
