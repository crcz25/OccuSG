#ifndef REGION_GEOMETRY_UTILS_HPP
#define REGION_GEOMETRY_UTILS_HPP

#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

/**
 * @brief Utilities for converting between pixel and map coordinates
 *
 * Handles the coordinate system conventions:
 * - OccupancyGrid: origin at bottom-left, X right, Y up
 * - OpenCV image: origin at top-left, X right, Y down (row-major)
 */
class CoordinateConverter {
public:
  /**
   * @brief Initialize converter with map metadata
   * @param resolution Map resolution in meters/pixel
   * @param width Map width in pixels
   * @param height Map height in pixels
   * @param origin_x Map origin X coordinate in meters
   * @param origin_y Map origin Y coordinate in meters
   */
  CoordinateConverter(double resolution, int width, int height, double origin_x,
                      double origin_y)
      : resolution_(resolution), width_(width), height_(height),
        origin_x_(origin_x), origin_y_(origin_y) {}

  /**
   * @brief Convert pixel coordinates to map frame (meters)
   * @param pixel_x Column index in image (0 = left edge)
   * @param pixel_y Row index in image (0 = bottom edge, already flipped)
   * @param map_x Output X coordinate in map frame (meters)
   * @param map_y Output Y coordinate in map frame (meters)
   *
   * Conversion formula:
   * - The input image has been flipped with cv::flip(img, img, 0)
   * - So pixel_y=0 is already at the bottom (matches map origin)
   * - No additional Y-flip needed
   */
  void pixelToMap(int pixel_x, int pixel_y, double &map_x,
                  double &map_y) const {
    // Direct conversion - image is already flipped to match map frame
    map_x = origin_x_ + pixel_x * resolution_;
    map_y = origin_y_ + pixel_y * resolution_;
  }

  /**
   * @brief Convert map frame coordinates to pixel indices
   * @param map_x X coordinate in map frame (meters)
   * @param map_y Y coordinate in map frame (meters)
   * @param pixel_x Output column index
   * @param pixel_y Output row index (in flipped image space)
   */
  void mapToPixel(double map_x, double map_y, int &pixel_x,
                  int &pixel_y) const {
    // Direct conversion - image is already flipped to match map frame
    pixel_x = static_cast<int>(std::round((map_x - origin_x_) / resolution_));
    pixel_y = static_cast<int>(std::round((map_y - origin_y_) / resolution_));
  }

private:
  double resolution_;
  int width_;
  int height_;
  double origin_x_;
  double origin_y_;
};

/**
 * @brief Extract geometric properties from region contours
 */
class RegionGeometryExtractor {
public:
  /**
   * @brief Compute convex hull of a contour
   */
  static std::vector<cv::Point>
  computeConvexHull(const std::vector<cv::Point> &contour) {
    std::vector<cv::Point> hull;
    if (contour.size() >= 3) {
      cv::convexHull(contour, hull);
    }
    return hull;
  }

  /**
   * @brief Compute area of a region (in pixels²)
   */
  static double computeArea(const std::vector<cv::Point> &contour) {
    if (contour.size() < 3)
      return 0.0;
    return std::abs(cv::contourArea(contour));
  }

  /**
   * @brief Compute centroid of a contour
   */
  static cv::Point computeCentroid(const std::vector<cv::Point> &contour) {
    if (contour.empty())
      return cv::Point(0, 0);

    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0.0)
      return contour[0]; // Fallback to first point

    return cv::Point(static_cast<int>(m.m10 / m.m00),
                     static_cast<int>(m.m01 / m.m00));
  }

  /**
   * @brief Compute Intersection over Union (IoU) between two contours
   */
  static double computeIoU(const std::vector<cv::Point> &contour1,
                           const std::vector<cv::Point> &contour2,
                           const cv::Size &image_size) {
    if (contour1.empty() || contour2.empty())
      return 0.0;

    // Create binary masks
    cv::Mat mask1 = cv::Mat::zeros(image_size, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(image_size, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours1 = {contour1};
    std::vector<std::vector<cv::Point>> contours2 = {contour2};

    cv::fillPoly(mask1, contours1, cv::Scalar(255));
    cv::fillPoly(mask2, contours2, cv::Scalar(255));

    // Compute intersection and union
    cv::Mat intersection, union_mat;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, union_mat);

    double intersection_area = cv::countNonZero(intersection);
    double union_area = cv::countNonZero(union_mat);

    if (union_area == 0.0)
      return 0.0;

    return intersection_area / union_area;
  }

  /**
   * @brief Check if two regions are adjacent (share a boundary)
   * Uses dilation to detect proximity
   */
  static bool areAdjacent(const std::vector<cv::Point> &contour1,
                          const std::vector<cv::Point> &contour2,
                          const cv::Size &image_size, int dilation_size = 2) {
    if (contour1.empty() || contour2.empty())
      return false;

    // Create masks
    cv::Mat mask1 = cv::Mat::zeros(image_size, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(image_size, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours1 = {contour1};
    std::vector<std::vector<cv::Point>> contours2 = {contour2};

    cv::fillPoly(mask1, contours1, cv::Scalar(255));
    cv::fillPoly(mask2, contours2, cv::Scalar(255));

    // Dilate first mask
    cv::Mat dilated;
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(dilation_size, dilation_size));
    cv::dilate(mask1, dilated, element);

    // Check overlap with second mask
    cv::Mat overlap;
    cv::bitwise_and(dilated, mask2, overlap);

    return cv::countNonZero(overlap) > 0;
  }
};

#endif // REGION_GEOMETRY_UTILS_HPP
