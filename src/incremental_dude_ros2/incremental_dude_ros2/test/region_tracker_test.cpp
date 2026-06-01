#include "basic_region_tracker.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

incremental_dude_msgs::msg::Region2D makeSquareRegion(
	const int id,
	const double min_x,
	const double min_y,
	const double max_x,
	const double max_y,
	const std::vector<int>& adjacent_ids = {})
{
	incremental_dude_msgs::msg::Region2D region;
	region.id = id;
	region.centroid.x = 0.5 * (min_x + max_x);
	region.centroid.y = 0.5 * (min_y + max_y);
	region.centroid.theta = 0.0;
	region.area = (max_x - min_x) * (max_y - min_y);
	region.adjacent_ids = adjacent_ids;

	const std::vector<std::pair<double, double>> points = {
		{min_x, min_y},
		{max_x, min_y},
		{max_x, max_y},
		{min_x, max_y},
	};
	for (const auto& point : points) {
		geometry_msgs::msg::Point32 msg_point;
		msg_point.x = static_cast<float>(point.first);
		msg_point.y = static_cast<float>(point.second);
		msg_point.z = 0.0f;
		region.polygon.points.push_back(msg_point);
		region.convex_hull.points.push_back(msg_point);
	}
	return region;
}

incremental_dude_msgs::msg::Region2DArray makeFrame(
	const std::vector<incremental_dude_msgs::msg::Region2D>& regions)
{
	incremental_dude_msgs::msg::Region2DArray frame;
	frame.regions = regions;
	return frame;
}

bool expectEqual(const int actual, const int expected, const std::string& label)
{
	if (actual == expected) {
		return true;
	}
	std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
	return false;
}

bool expectVectorEqual(
	const std::vector<int>& actual,
	const std::vector<int>& expected,
	const std::string& label)
{
	if (actual == expected) {
		return true;
	}
	std::cerr << label << ": expected [";
	for (size_t i = 0; i < expected.size(); ++i) {
		std::cerr << (i == 0 ? "" : ", ") << expected[i];
	}
	std::cerr << "], got [";
	for (size_t i = 0; i < actual.size(); ++i) {
		std::cerr << (i == 0 ? "" : ", ") << actual[i];
	}
	std::cerr << "]\n";
	return false;
}

int runTrackerTest()
{
	BasicRegionTracker::Config config;
	config.min_iou = 0.20;
	config.max_centroid_distance = 1.0;
	config.min_area_ratio = 0.25;
	config.max_area_ratio = 4.0;
	config.max_missed_frames = 3;

	BasicRegionTracker tracker(config);

	const auto first = tracker.canonicalize(makeFrame({
		makeSquareRegion(0, 0.0, 0.0, 2.0, 2.0, {1}),
		makeSquareRegion(1, 10.0, 0.0, 12.0, 2.0, {0}),
	}));
	if (!expectEqual(first.message.regions[0].id, 1, "first left canonical id") ||
		!expectEqual(first.message.regions[1].id, 2, "first right canonical id")) {
		return 1;
	}
	if (!expectVectorEqual(first.message.regions[0].adjacent_ids, {2}, "first left adjacency") ||
		!expectVectorEqual(first.message.regions[1].adjacent_ids, {1}, "first right adjacency")) {
		return 2;
	}

	const auto swapped = tracker.canonicalize(makeFrame({
		makeSquareRegion(0, 10.1, 0.0, 12.1, 2.0, {1}),
		makeSquareRegion(1, 0.1, 0.0, 2.1, 2.0, {0}),
	}));
	if (!expectEqual(swapped.message.regions[0].id, 2, "swapped right canonical id") ||
		!expectEqual(swapped.message.regions[1].id, 1, "swapped left canonical id")) {
		return 3;
	}
	if (!expectVectorEqual(swapped.message.regions[0].adjacent_ids, {1}, "swapped right adjacency") ||
		!expectVectorEqual(swapped.message.regions[1].adjacent_ids, {2}, "swapped left adjacency")) {
		return 4;
	}

	const auto with_new_region = tracker.canonicalize(makeFrame({
		makeSquareRegion(0, 0.2, 0.0, 2.2, 2.0),
		makeSquareRegion(1, 10.2, 0.0, 12.2, 2.0),
		makeSquareRegion(2, 20.0, 0.0, 22.0, 2.0),
	}));
	if (!expectEqual(with_new_region.message.regions[0].id, 1, "with-new left canonical id") ||
		!expectEqual(with_new_region.message.regions[1].id, 2, "with-new right canonical id") ||
		!expectEqual(with_new_region.message.regions[2].id, 3, "with-new third canonical id")) {
		return 5;
	}

	const auto missing_right = tracker.canonicalize(makeFrame({
		makeSquareRegion(0, 0.3, 0.0, 2.3, 2.0),
	}));
	if (!expectEqual(missing_right.message.regions[0].id, 1, "missing-right left canonical id")) {
		return 6;
	}

	const auto empty = tracker.canonicalize(makeFrame({}));
	if (!empty.message.regions.empty()) {
		std::cerr << "empty frame should publish zero regions\n";
		return 7;
	}

	const auto right_reappears = tracker.canonicalize(makeFrame({
		makeSquareRegion(0, 10.3, 0.0, 12.3, 2.0),
		makeSquareRegion(1, 0.4, 0.0, 2.4, 2.0),
	}));
	if (!expectEqual(right_reappears.message.regions[0].id, 2, "reappeared right canonical id") ||
		!expectEqual(right_reappears.message.regions[1].id, 1, "reappeared left canonical id")) {
		return 8;
	}

	return 0;
}

}  // namespace

int main()
{
	return runTrackerTest();
}
