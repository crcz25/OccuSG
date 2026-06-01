#include "inc_decomp.hpp"

#include <rclcpp/rclcpp.hpp>

#include <cstdlib>

namespace {

cv::Mat makeFrame(int width, int height, const cv::Rect &free_rect) {
  cv::Mat image = cv::Mat::zeros(height, width, CV_8UC1);
  cv::rectangle(image, free_rect, cv::Scalar(255), cv::FILLED);
  return image;
}

bool hasConsistentGraph(const Stable_graph &graph) {
  return graph.Region_contour.size() == graph.Region_centroid.size();
}

int runRegression() {
  Incremental_Decomposer decomposer;
  const float resolution = 0.05f;
  const float pixel_tau = 3.0f / resolution;

  const cv::Mat frame_t_minus_1 = makeFrame(60, 50, cv::Rect(10, 10, 30, 20));
  const Stable_graph &first =
      decomposer.decompose_image(frame_t_minus_1, pixel_tau, cv::Point2f(0.0f, 0.0f), resolution);
  if (first.Region_contour.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Expected non-empty decomposition for first frame");
    return 1;
  }
  if (!hasConsistentGraph(first)) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "First frame produced inconsistent contour/centroid counts");
    return 4;
  }

  const cv::Mat frame_t = makeFrame(60, 51, cv::Rect(10, 10, 30, 20));
  const Stable_graph &second =
      decomposer.decompose_image(frame_t, pixel_tau, cv::Point2f(0.0f, 0.0f), resolution);
  if (second.Region_contour.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Regression reproduced: differential-empty frame produced zero regions");
    return 2;
  }
  if (!hasConsistentGraph(second)) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Second frame produced inconsistent contour/centroid counts");
    return 5;
  }

  if (decomposer.last_branch_ != "accept_reuse_adjusted_no_differential" &&
      decomposer.last_branch_ != "accept_incremental" &&
      decomposer.last_branch_ != "fallback_full_decomposition") {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Unexpected branch: %s", decomposer.last_branch_.c_str());
    return 3;
  }

  Incremental_Decomposer degenerate_decomposer;
  const cv::Mat one_pixel = makeFrame(20, 20, cv::Rect(5, 5, 1, 1));
  const Stable_graph &degenerate =
      degenerate_decomposer.decompose_image(one_pixel, pixel_tau, cv::Point2f(0.0f, 0.0f), resolution);
  if (!degenerate.Region_contour.empty() || !hasConsistentGraph(degenerate)) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Degenerate one-pixel frame should be rejected as empty-valid output");
    return 6;
  }

  Incremental_Decomposer corrupted_state_decomposer;
  corrupted_state_decomposer.first_time = false;
  corrupted_state_decomposer.current_origin_ = cv::Point2f(0.0f, 0.0f);
  corrupted_state_decomposer.previous_rect = cv::Rect(10, 10, 30, 20);
  corrupted_state_decomposer.Stable.image_size = cv::Size(60, 50);
  corrupted_state_decomposer.Stable.Region_contour.push_back(
      {cv::Point(10, 10), cv::Point(40, 10), cv::Point(40, 30), cv::Point(10, 30)});
  const cv::Mat corrupted_frame = makeFrame(60, 50, cv::Rect(10, 10, 31, 20));
  const Stable_graph &repaired =
      corrupted_state_decomposer.decompose_image(corrupted_frame, pixel_tau, cv::Point2f(0.0f, 0.0f), resolution);
  if (!hasConsistentGraph(repaired)) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Corrupted previous state was not repaired");
    return 7;
  }

  Incremental_Decomposer expanded_empty_decomposer;
  expanded_empty_decomposer.first_time = false;
  expanded_empty_decomposer.current_origin_ = cv::Point2f(0.0f, 0.0f);
  expanded_empty_decomposer.previous_rect = cv::Rect(2, 2, 8, 8);
  expanded_empty_decomposer.Stable.image_size = cv::Size(80, 80);
  expanded_empty_decomposer.Stable.Region_contour.push_back(
      {cv::Point(2, 2), cv::Point(10, 2), cv::Point(10, 10), cv::Point(2, 10)});
  expanded_empty_decomposer.Stable.Region_centroid.push_back(cv::Point(6, 6));
  const cv::Mat disconnected_frame = makeFrame(80, 80, cv::Rect(50, 50, 20, 20));
  const Stable_graph &preserved =
      expanded_empty_decomposer.decompose_image(disconnected_frame, pixel_tau, cv::Point2f(0.0f, 0.0f), resolution);
  if (expanded_empty_decomposer.last_branch_ != "reject_expanded_empty_preserve_previous" ||
      preserved.Region_contour.empty() ||
      !hasConsistentGraph(preserved)) {
    RCLCPP_ERROR(rclcpp::get_logger("inc_decomp_regression_test"),
                 "Expanded-empty fallback did not preserve previous valid state safely; branch=%s",
                 expanded_empty_decomposer.last_branch_.c_str());
    return 8;
  }

  RCLCPP_INFO(rclcpp::get_logger("inc_decomp_regression_test"),
              "Regression test passed with %zu regions on dimension-change frame using branch %s",
              second.Region_contour.size(), decomposer.last_branch_.c_str());
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  const int rc = runRegression();
  rclcpp::shutdown();
  return rc;
}
