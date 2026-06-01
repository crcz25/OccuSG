#ifndef BASIC_REGION_TRACKER_HPP
#define BASIC_REGION_TRACKER_HPP

#include "incremental_dude_msgs/msg/region2_d.hpp"
#include "incremental_dude_msgs/msg/region2_d_array.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

class BasicRegionTracker
{
public:
	struct Config
	{
		double min_iou{0.20};
		double max_centroid_distance{1.0};
		double min_area_ratio{0.25};
		double max_area_ratio{4.0};
		int max_missed_frames{3};
		bool publish_debug{false};
	};

	struct Stats
	{
		size_t matched_regions{0};
		size_t new_regions{0};
		size_t removed_tracks{0};
		size_t active_tracks{0};
	};

	struct Result
	{
		incremental_dude_msgs::msg::Region2DArray message;
		Stats stats;
	};

	BasicRegionTracker() = default;
	explicit BasicRegionTracker(const Config& config)
	: config_(config)
	{
		normalizeConfig();
	}

	void setConfig(const Config& config)
	{
		config_ = config;
		normalizeConfig();
	}

	const Config& config() const
	{
		return config_;
	}

	void reset()
	{
		tracks_.clear();
		next_canonical_id_ = 1;
		frame_index_ = 0;
	}

	Result canonicalize(const incremental_dude_msgs::msg::Region2DArray& input)
	{
		Result result;
		result.message = input;
		++frame_index_;

		std::vector<RegionFeatures> current_features;
		current_features.reserve(input.regions.size());
		for (const auto& region : input.regions) {
			current_features.push_back(extractFeatures(region));
		}

		std::vector<CandidateMatch> candidates;
		candidates.reserve(tracks_.size() * current_features.size());
		for (size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
			const auto& track = tracks_[track_index];
			if (!track.features.valid) {
				continue;
			}
			for (size_t current_index = 0; current_index < current_features.size(); ++current_index) {
				const auto candidate = scoreCandidate(track_index, current_index, track.features, current_features[current_index]);
				if (candidate.accepted) {
					candidates.push_back(candidate);
				}
			}
		}

		std::sort(
			candidates.begin(),
			candidates.end(),
			[](const CandidateMatch& lhs, const CandidateMatch& rhs) {
				if (lhs.score != rhs.score) {
					return lhs.score > rhs.score;
				}
				if (lhs.iou != rhs.iou) {
					return lhs.iou > rhs.iou;
				}
				if (lhs.centroid_distance != rhs.centroid_distance) {
					return lhs.centroid_distance < rhs.centroid_distance;
				}
				if (lhs.canonical_id != rhs.canonical_id) {
					return lhs.canonical_id < rhs.canonical_id;
				}
				return lhs.current_index < rhs.current_index;
			});

		std::vector<bool> track_matched(tracks_.size(), false);
		std::vector<bool> current_matched(current_features.size(), false);
		std::vector<int> canonical_ids(current_features.size(), -1);

		for (const auto& candidate : candidates) {
			if (track_matched[candidate.track_index] || current_matched[candidate.current_index]) {
				continue;
			}

			auto& track = tracks_[candidate.track_index];
			track.features = current_features[candidate.current_index];
			track.missed_frames = 0;
			track.last_seen_frame = frame_index_;

			track_matched[candidate.track_index] = true;
			current_matched[candidate.current_index] = true;
			canonical_ids[candidate.current_index] = track.canonical_id;
			++result.stats.matched_regions;
		}

		for (size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
			if (!track_matched[track_index]) {
				++tracks_[track_index].missed_frames;
			}
		}

		for (size_t current_index = 0; current_index < current_features.size(); ++current_index) {
			if (current_matched[current_index]) {
				continue;
			}
			const int canonical_id = next_canonical_id_++;
			canonical_ids[current_index] = canonical_id;
			if (current_features[current_index].valid) {
				TrackedRegion track;
				track.canonical_id = canonical_id;
				track.features = current_features[current_index];
				track.last_seen_frame = frame_index_;
				tracks_.push_back(track);
			}
			++result.stats.new_regions;
		}

		const size_t tracks_before_prune = tracks_.size();
		tracks_.erase(
			std::remove_if(
				tracks_.begin(),
				tracks_.end(),
				[this](const TrackedRegion& track) {
					return track.missed_frames > config_.max_missed_frames;
				}),
			tracks_.end());
		result.stats.removed_tracks = tracks_before_prune - tracks_.size();
		result.stats.active_tracks = tracks_.size();

		applyCanonicalIds(input, canonical_ids, result.message);
		return result;
	}

private:
	struct Bounds
	{
		double min_x{std::numeric_limits<double>::infinity()};
		double min_y{std::numeric_limits<double>::infinity()};
		double max_x{-std::numeric_limits<double>::infinity()};
		double max_y{-std::numeric_limits<double>::infinity()};
		bool valid{false};
	};

	struct RegionFeatures
	{
		int frame_local_id{-1};
		double centroid_x{0.0};
		double centroid_y{0.0};
		double area{0.0};
		Bounds bounds;
		bool valid{false};
	};

	struct TrackedRegion
	{
		int canonical_id{0};
		RegionFeatures features;
		int missed_frames{0};
		uint64_t last_seen_frame{0};
	};

	struct CandidateMatch
	{
		size_t track_index{0};
		size_t current_index{0};
		int canonical_id{0};
		double score{0.0};
		double iou{0.0};
		double centroid_distance{0.0};
		bool accepted{false};
	};

	static bool isFinite(const double value)
	{
		return std::isfinite(value);
	}

	static Bounds boundsFromPolygon(const geometry_msgs::msg::Polygon& polygon)
	{
		Bounds bounds;
		if (polygon.points.empty()) {
			return bounds;
		}

		for (const auto& point : polygon.points) {
			const double x = static_cast<double>(point.x);
			const double y = static_cast<double>(point.y);
			if (!isFinite(x) || !isFinite(y)) {
				continue;
			}
			bounds.min_x = std::min(bounds.min_x, x);
			bounds.min_y = std::min(bounds.min_y, y);
			bounds.max_x = std::max(bounds.max_x, x);
			bounds.max_y = std::max(bounds.max_y, y);
			bounds.valid = true;
		}
		return bounds;
	}

	static double polygonArea(const geometry_msgs::msg::Polygon& polygon)
	{
		if (polygon.points.size() < 3) {
			return 0.0;
		}

		double signed_area = 0.0;
		for (size_t i = 0; i < polygon.points.size(); ++i) {
			const auto& a = polygon.points[i];
			const auto& b = polygon.points[(i + 1) % polygon.points.size()];
			signed_area += static_cast<double>(a.x) * static_cast<double>(b.y) -
				static_cast<double>(b.x) * static_cast<double>(a.y);
		}
		return std::abs(0.5 * signed_area);
	}

	static double bboxIoU(const Bounds& lhs, const Bounds& rhs)
	{
		if (!lhs.valid || !rhs.valid) {
			return 0.0;
		}

		const double intersection_width =
			std::max(0.0, std::min(lhs.max_x, rhs.max_x) - std::max(lhs.min_x, rhs.min_x));
		const double intersection_height =
			std::max(0.0, std::min(lhs.max_y, rhs.max_y) - std::max(lhs.min_y, rhs.min_y));
		const double intersection_area = intersection_width * intersection_height;
		const double lhs_area = std::max(0.0, lhs.max_x - lhs.min_x) * std::max(0.0, lhs.max_y - lhs.min_y);
		const double rhs_area = std::max(0.0, rhs.max_x - rhs.min_x) * std::max(0.0, rhs.max_y - rhs.min_y);
		const double union_area = lhs_area + rhs_area - intersection_area;
		if (union_area <= 0.0) {
			return 0.0;
		}
		return intersection_area / union_area;
	}

	static double distance(const RegionFeatures& lhs, const RegionFeatures& rhs)
	{
		const double dx = lhs.centroid_x - rhs.centroid_x;
		const double dy = lhs.centroid_y - rhs.centroid_y;
		return std::sqrt(dx * dx + dy * dy);
	}

	RegionFeatures extractFeatures(const incremental_dude_msgs::msg::Region2D& region) const
	{
		RegionFeatures features;
		features.frame_local_id = region.id;
		features.centroid_x = region.centroid.x;
		features.centroid_y = region.centroid.y;
		features.area = region.area > 0.0 ? static_cast<double>(region.area) : polygonArea(region.polygon);
		features.bounds =
			region.convex_hull.points.size() >= 3 ? boundsFromPolygon(region.convex_hull) : boundsFromPolygon(region.polygon);
		features.valid =
			isFinite(features.centroid_x) &&
			isFinite(features.centroid_y) &&
			isFinite(features.area) &&
			features.area > 0.0 &&
			features.bounds.valid;
		return features;
	}

	CandidateMatch scoreCandidate(
		const size_t track_index,
		const size_t current_index,
		const RegionFeatures& previous,
		const RegionFeatures& current) const
	{
		CandidateMatch candidate;
		candidate.track_index = track_index;
		candidate.current_index = current_index;
		candidate.canonical_id = tracks_[track_index].canonical_id;

		if (!previous.valid || !current.valid || previous.area <= 0.0 || current.area <= 0.0) {
			return candidate;
		}

		const double area_ratio = current.area / previous.area;
		const bool area_ratio_ok =
			area_ratio >= config_.min_area_ratio &&
			area_ratio <= config_.max_area_ratio;
		candidate.iou = bboxIoU(previous.bounds, current.bounds);
		candidate.centroid_distance = distance(previous, current);
		const bool geometry_gate_ok =
			candidate.iou >= config_.min_iou ||
			candidate.centroid_distance <= config_.max_centroid_distance;

		if (!area_ratio_ok || !geometry_gate_ok) {
			return candidate;
		}

		const double centroid_score =
			config_.max_centroid_distance > 0.0 ?
			std::max(0.0, 1.0 - (candidate.centroid_distance / config_.max_centroid_distance)) :
			0.0;
		const double area_score = std::min(area_ratio, 1.0 / area_ratio);
		candidate.score = (2.0 * candidate.iou) + centroid_score + (0.25 * area_score);
		candidate.accepted = true;
		return candidate;
	}

	static void applyCanonicalIds(
		const incremental_dude_msgs::msg::Region2DArray& input,
		const std::vector<int>& canonical_ids,
		incremental_dude_msgs::msg::Region2DArray& output)
	{
		std::map<int, int> frame_to_canonical;
		for (size_t i = 0; i < input.regions.size() && i < canonical_ids.size(); ++i) {
			frame_to_canonical[input.regions[i].id] = canonical_ids[i];
		}

		for (size_t i = 0; i < output.regions.size() && i < canonical_ids.size(); ++i) {
			output.regions[i].id = canonical_ids[i];

			std::vector<int> remapped_adjacency;
			remapped_adjacency.reserve(input.regions[i].adjacent_ids.size());
			for (const int frame_local_adjacent_id : input.regions[i].adjacent_ids) {
				const auto found = frame_to_canonical.find(frame_local_adjacent_id);
				if (found != frame_to_canonical.end()) {
					remapped_adjacency.push_back(found->second);
				}
			}
			std::sort(remapped_adjacency.begin(), remapped_adjacency.end());
			remapped_adjacency.erase(
				std::unique(remapped_adjacency.begin(), remapped_adjacency.end()),
				remapped_adjacency.end());
			output.regions[i].adjacent_ids = remapped_adjacency;
		}
	}

	void normalizeConfig()
	{
		config_.min_iou = std::max(0.0, std::min(1.0, config_.min_iou));
		config_.max_centroid_distance = std::max(0.0, config_.max_centroid_distance);
		config_.min_area_ratio = std::max(0.0, config_.min_area_ratio);
		config_.max_area_ratio = std::max(config_.min_area_ratio, config_.max_area_ratio);
		config_.max_missed_frames = std::max(0, config_.max_missed_frames);
	}

	Config config_;
	std::vector<TrackedRegion> tracks_;
	int next_canonical_id_{1};
	uint64_t frame_index_{0};
};

#endif // BASIC_REGION_TRACKER_HPP
