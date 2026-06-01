//ROS2
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "std_msgs/msg/string.hpp"
#include "rclcpp/rclcpp.hpp"

// Region messages
#include "incremental_dude_msgs/msg/region2_d.hpp"
#include "incremental_dude_msgs/msg/region2_d_array.hpp"

//openCV
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

//DuDe
#include "basic_region_tracker.hpp"
#include "inc_decomp.hpp"
#include "region_geometry_utils.hpp"
#include "region_ros_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
class RuntimeProfiler
{
public:
	RuntimeProfiler(
		const std::string& node_name,
		const std::string& package_name,
		const std::string& file_tag,
		bool enabled,
		const std::string& output_path,
		const std::string& run_name,
		bool save_on_shutdown,
		int discard_first_n)
	: node_name_(node_name),
	  package_name_(package_name),
	  file_tag_(file_tag),
	  enabled_(enabled),
	  output_path_(output_path),
	  run_name_(run_name),
	  save_on_shutdown_(save_on_shutdown),
	  discard_first_n_(std::max(0, discard_first_n)),
	  started_at_(timestamp())
	{}

	~RuntimeProfiler()
	{
		if (save_on_shutdown_) {
			save();
		}
	}

	void record(const std::string& stage_name, double elapsed_ms)
	{
		if (!enabled_ || !std::isfinite(elapsed_ms)) {
			return;
		}
		samples_[stage_name].push_back(elapsed_ms);
	}

	void save()
	{
		if (!enabled_) {
			return;
		}
		const std::string base_path = output_path_.empty() ? "." : output_path_;
		std::filesystem::create_directories(base_path);
		const std::string path = base_path + "/" + run_name_ + "." + file_tag_ + ".json";
		std::ofstream out(path);
		if (!out) {
			return;
		}
		ended_at_ = timestamp();
		out << "{\n";
		out << "  \"run_name\": " << jsonString(run_name_) << ",\n";
		out << "  \"node_name\": " << jsonString(node_name_) << ",\n";
		out << "  \"package_name\": " << jsonString(package_name_) << ",\n";
		out << "  \"discarded_warmup_count\": " << discard_first_n_ << ",\n";
		out << "  \"started_at\": " << jsonString(started_at_) << ",\n";
		out << "  \"ended_at\": " << jsonString(ended_at_) << ",\n";
		out << "  \"metadata\": " << metadataJson() << ",\n";
		out << "  \"stages\": {\n";
		size_t stage_index = 0;
		for (const auto& entry : samples_) {
			out << "    " << jsonString(entry.first) << ": {\n";
			out << "      \"samples_ms\": [";
			for (size_t i = 0; i < entry.second.size(); ++i) {
				if (i > 0) {
					out << ", ";
				}
				out << std::fixed << std::setprecision(6) << entry.second[i];
			}
			out << "],\n";
			out << "      \"summary\": " << summaryJson(entry.second) << "\n";
			out << "    }" << (++stage_index < samples_.size() ? "," : "") << "\n";
		}
		out << "  }\n";
		out << "}\n";
	}

private:
	static std::string timestamp()
	{
		const auto now = std::time(nullptr);
		std::tm tm {};
		gmtime_r(&now, &tm);
		char buffer[32];
		std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
		return std::string(buffer);
	}

	static std::string jsonString(const std::string& value)
	{
		std::ostringstream out;
		out << '"';
		for (const char c : value) {
			switch (c) {
			case '"': out << "\\\""; break;
			case '\\': out << "\\\\"; break;
			case '\n': out << "\\n"; break;
			case '\r': out << "\\r"; break;
			case '\t': out << "\\t"; break;
			default: out << c; break;
			}
		}
		out << '"';
		return out.str();
	}

	static std::string firstCpuModel()
	{
		std::ifstream in("/proc/cpuinfo");
		std::string line;
		while (std::getline(in, line)) {
			const std::string key = "model name";
			if (line.compare(0, key.size(), key) == 0) {
				const auto pos = line.find(':');
				if (pos != std::string::npos) {
					return line.substr(pos + 2);
				}
			}
		}
		return "";
	}

	static double ramGb()
	{
		std::ifstream in("/proc/meminfo");
		std::string key;
		std::string unit;
		double kb = 0.0;
		while (in >> key >> kb >> unit) {
			if (key == "MemTotal:") {
				return kb / (1024.0 * 1024.0);
			}
		}
		return 0.0;
	}

	std::string metadataJson() const
	{
		char hostname[256] = "";
		gethostname(hostname, sizeof(hostname) - 1);
		const char* ros_distro = std::getenv("ROS_DISTRO");
		std::ostringstream out;
		out << "{";
		out << "\"hostname\": " << jsonString(hostname) << ", ";
		out << "\"cpu_model\": " << jsonString(firstCpuModel()) << ", ";
		out << "\"logical_cores\": " << std::thread::hardware_concurrency() << ", ";
		out << "\"ram_gb\": " << std::fixed << std::setprecision(3) << ramGb() << ", ";
		out << "\"ros_distro\": " << jsonString(ros_distro ? ros_distro : "") << ", ";
		out << "\"date_time\": " << jsonString(timestamp()) << ", ";
		out << "\"dude_decomposition_definition\": "
			<< jsonString("accepted decomposition updates only") << ", ";
		out << "\"region_tracking_definition\": "
			<< jsonString("canonicalize(msg), recorded only when tracker is enabled");
		out << "}";
		return out.str();
	}

	std::string summaryJson(const std::vector<double>& raw_values) const
	{
		std::vector<double> values;
		for (size_t i = static_cast<size_t>(discard_first_n_); i < raw_values.size(); ++i) {
			values.push_back(raw_values[i]);
		}
		if (values.empty()) {
			return "{\"n\": 0, \"mean_ms\": null, \"std_ms\": null, \"min_ms\": null, \"max_ms\": null}";
		}
		double sum = 0.0;
		double min_value = values.front();
		double max_value = values.front();
		for (const double value : values) {
			sum += value;
			min_value = std::min(min_value, value);
			max_value = std::max(max_value, value);
		}
		const double mean = sum / static_cast<double>(values.size());
		double variance = 0.0;
		for (const double value : values) {
			variance += (value - mean) * (value - mean);
		}
		variance /= static_cast<double>(values.size());
		std::ostringstream out;
		out << std::fixed << std::setprecision(6)
			<< "{\"n\": " << values.size()
			<< ", \"mean_ms\": " << mean
			<< ", \"std_ms\": " << std::sqrt(std::max(0.0, variance))
			<< ", \"min_ms\": " << min_value
			<< ", \"max_ms\": " << max_value << "}";
		return out.str();
	}

	std::string node_name_;
	std::string package_name_;
	std::string file_tag_;
	bool enabled_{false};
	std::string output_path_;
	std::string run_name_;
	bool save_on_shutdown_{true};
	int discard_first_n_{0};
	std::string started_at_;
	std::string ended_at_;
	std::map<std::string, std::vector<double>> samples_;
};
} // namespace

class ROS_handler : public rclcpp::Node
{
	image_transport::Publisher image_pub_;
	std::shared_ptr<image_transport::ImageTransport> image_transport_;
	cv_bridge::CvImagePtr cv_ptr_;

	rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr chat_sub_;
	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Publisher<incremental_dude_msgs::msg::Region2DArray>::SharedPtr region_pub_;

	float Decomp_threshold_;
	std::string occupancy_grid_topic_;
	bool region_tracker_enable_{true};
	bool region_tracker_publish_debug_{false};
	bool preserve_previous_on_invalid_update_{true};
	bool publish_stale_regions_on_reject_{true};
	int min_valid_region_count_{1};
	double max_unknown_ratio_for_update_{0.98};
	double max_region_area_drop_ratio_{0.85};
	double max_region_count_drop_ratio_{0.85};
	Incremental_Decomposer inc_decomp_;
	Stable_graph Stable_;
	Stable_graph last_valid_stable_;
	BasicRegionTracker region_tracker_;
	incremental_dude_msgs::msg::Region2DArray last_valid_region_msg_;
	bool has_last_valid_region_msg_{false};
	size_t last_valid_region_count_{0};
	double last_valid_region_area_{0.0};
	size_t rejected_region_updates_{0};

	cv::Mat image2save_clean_;
	cv::Mat image2save_black_;
	cv::Mat image2save_Inc_;

	size_t processed_maps_{0};
	double cumulative_whole_time_ms_{0.0};
	double cumulative_squared_whole_time_ms_{0.0};
	size_t profiling_samples_since_save_{0};
	std::unique_ptr<RuntimeProfiler> profiler_;

public:
	ROS_handler()
	: Node("incremental_decomposer")
	{
		this->declare_parameter("decomp_threshold", 3.0);
		this->declare_parameter("occupancy_grid_topic", "/mapUAV");
		region_tracker_enable_ =
			this->declare_parameter<bool>("region_tracker_enable", true);
		BasicRegionTracker::Config tracker_config;
		tracker_config.min_iou =
			this->declare_parameter<double>("region_tracker_min_iou", 0.20);
		tracker_config.max_centroid_distance =
			this->declare_parameter<double>("region_tracker_max_centroid_distance", 1.0);
		tracker_config.min_area_ratio =
			this->declare_parameter<double>("region_tracker_min_area_ratio", 0.25);
		tracker_config.max_area_ratio =
			this->declare_parameter<double>("region_tracker_max_area_ratio", 4.0);
		tracker_config.max_missed_frames =
			this->declare_parameter<int>("region_tracker_max_missed_frames", 3);
		tracker_config.publish_debug =
			this->declare_parameter<bool>("region_tracker_publish_debug", false);
		region_tracker_publish_debug_ = tracker_config.publish_debug;
		region_tracker_.setConfig(tracker_config);
		preserve_previous_on_invalid_update_ =
			this->declare_parameter<bool>("preserve_previous_on_invalid_update", true);
		publish_stale_regions_on_reject_ =
			this->declare_parameter<bool>("publish_stale_regions_on_reject", true);
		min_valid_region_count_ =
			this->declare_parameter<int>("min_valid_region_count", 1);
		max_unknown_ratio_for_update_ =
			this->declare_parameter<double>("max_unknown_ratio_for_update", 0.98);
		max_region_area_drop_ratio_ =
			this->declare_parameter<double>("max_region_area_drop_ratio", 0.85);
		max_region_count_drop_ratio_ =
			this->declare_parameter<double>("max_region_count_drop_ratio", 0.85);
		const bool enable_profiling =
			this->declare_parameter<bool>("enable_profiling", false);
		const std::string profiling_output_path =
			this->declare_parameter<std::string>("profiling_output_path", "");
		const std::string profiling_run_name =
			this->declare_parameter<std::string>("profiling_run_name", "run");
		const bool profiling_save_on_shutdown =
			this->declare_parameter<bool>("profiling_save_on_shutdown", true);
		const int profiling_discard_first_n =
			this->declare_parameter<int>("profiling_discard_first_n", 5);
		profiler_ = std::make_unique<RuntimeProfiler>(
			"incremental_decomposer",
			"incremental_dude_ros2",
			"incremental_dude",
			enable_profiling,
			profiling_output_path,
			profiling_run_name,
			profiling_save_on_shutdown,
			profiling_discard_first_n);
		if (enable_profiling) {
			RCLCPP_INFO(
				this->get_logger(),
				"Runtime profiling enabled: %s/%s.incremental_dude.json",
				profiling_output_path.c_str(),
				profiling_run_name.c_str());
		}

		Decomp_threshold_ =
			static_cast<float>(this->get_parameter("decomp_threshold").as_double());
		occupancy_grid_topic_ =
			this->get_parameter("occupancy_grid_topic").as_string();

		RCLCPP_INFO(
			this->get_logger(),
			"Waiting for the map on topic '%s'",
			occupancy_grid_topic_.c_str());
		RCLCPP_INFO(
			this->get_logger(),
			"Region tracker %s: bbox_iou>=%.2f or centroid<=%.2fm, area_ratio=[%.2f, %.2f], max_missed=%d",
			region_tracker_enable_ ? "enabled" : "disabled",
			region_tracker_.config().min_iou,
			region_tracker_.config().max_centroid_distance,
			region_tracker_.config().min_area_ratio,
			region_tracker_.config().max_area_ratio,
			region_tracker_.config().max_missed_frames);
		RCLCPP_INFO(
			this->get_logger(),
			"Region update guard: preserve_previous=%s publish_stale=%s min_regions=%d max_unknown=%.2f max_area_drop=%.2f max_count_drop=%.2f",
			preserve_previous_on_invalid_update_ ? "true" : "false",
			publish_stale_regions_on_reject_ ? "true" : "false",
			min_valid_region_count_,
			max_unknown_ratio_for_update_,
			max_region_area_drop_ratio_,
			max_region_count_drop_ratio_);

		map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
			occupancy_grid_topic_, rclcpp::QoS(2),
			std::bind(&ROS_handler::mapCallback, this, std::placeholders::_1));
		chat_sub_ = this->create_subscription<std_msgs::msg::String>(
			"chatter", rclcpp::QoS(1),
			std::bind(&ROS_handler::chatCallback, this, std::placeholders::_1));
		timer_ = this->create_wall_timer(
			std::chrono::milliseconds(500),
			std::bind(&ROS_handler::metronomeCallback, this));
	}

	void init_publishers()
	{
		image_transport_ =
			std::make_shared<image_transport::ImageTransport>(shared_from_this());
		image_pub_ = image_transport_->advertise("/dude/tagged_image", 1);
		region_pub_ = this->create_publisher<incremental_dude_msgs::msg::Region2DArray>(
			"/dude/regions", rclcpp::QoS(1));

		cv_ptr_ = std::make_shared<cv_bridge::CvImage>();
		cv_ptr_->encoding = sensor_msgs::image_encodings::RGB8;
	}

	void saveProfiling()
	{
		if (profiler_) {
			profiler_->save();
		}
	}

	void maybeSaveProfilingSnapshot()
	{
		if (!profiler_) {
			return;
		}
		++profiling_samples_since_save_;
		if (profiling_samples_since_save_ < 25) {
			return;
		}
		profiler_->save();
		profiling_samples_since_save_ = 0;
	}

private:
	struct PublishedContour
	{
		int id;
		const std::vector<cv::Point>* contour;
	};

	struct RegionUpdateValidation
	{
		bool accepted{true};
		std::string reason{"accepted"};
		size_t region_count{0};
		double total_area{0.0};
		double unknown_ratio{0.0};
		double free_ratio{0.0};
	};

	double getTime() const
	{
		return std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr map)
	{
		double begin_process, end_process, begin_whole, occupancy_time, decompose_time, drawPublish_time, whole_time;
		begin_whole = begin_process = getTime();

		size_t free_cells = 0;
		size_t occupied_cells = 0;
		size_t unknown_cells = 0;
		for (const auto value : map->data) {
			if (value < 0) {
				++unknown_cells;
			} else if (value >= 90) {
				++occupied_cells;
			} else if (value < 10) {
				++free_cells;
			}
		}
		RCLCPP_DEBUG_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Map stats size=%ux%u resolution=%.3f origin=(%.3f, %.3f) cells free=%zu occupied=%zu unknown=%zu",
			map->info.width,
			map->info.height,
			map->info.resolution,
			map->info.origin.position.x,
			map->info.origin.position.y,
			free_cells,
			occupied_cells,
			unknown_cells);

		const size_t expected_cells =
			static_cast<size_t>(map->info.width) * static_cast<size_t>(map->info.height);
		if (map->info.width == 0 || map->info.height == 0 || expected_cells == 0) {
			RCLCPP_WARN(
				this->get_logger(),
				"Skipping empty occupancy grid (%u x %u).",
				map->info.width,
				map->info.height);
			reject_region_update(map, "empty_occupancy_grid");
			return;
		}
		if (map->info.resolution <= 0.0f) {
			RCLCPP_WARN(
				this->get_logger(),
				"Skipping occupancy grid with invalid resolution %.6f.",
				map->info.resolution);
			reject_region_update(map, "invalid_occupancy_grid_resolution");
			return;
		}
		if (map->data.size() < expected_cells) {
			RCLCPP_WARN(
				this->get_logger(),
				"Skipping occupancy grid with %zu cells, expected at least %zu.",
				map->data.size(),
				expected_cells);
			reject_region_update(map, "occupancy_grid_data_too_short");
			return;
		}

	///////////////////////Occupancy to clean image
		cv::Mat grad, img(map->info.height, map->info.width, CV_8U);
		img.data = (unsigned char *)(&(map->data[0]));
		cv::Mat received_image = img.clone();
		if (received_image.empty()) {
			RCLCPP_WARN(this->get_logger(), "Skipping occupancy grid because the converted image is empty.");
			reject_region_update(map, "converted_image_empty");
			return;
		}

		float pixel_Tau = Decomp_threshold_ / map->info.resolution;
		cv_ptr_->header = map->header;
		cv::Point2f origin = cv::Point2f(map->info.origin.position.x, map->info.origin.position.y);

		cv::Rect first_rect = find_image_bounding_Rect(received_image);
		if (first_rect.width <= 0 || first_rect.height <= 0) {
			RCLCPP_WARN(this->get_logger(), "Skipping occupancy grid because the valid-image bounding box is empty.");
			reject_region_update(map, "valid_image_bounding_box_empty");
			return;
		}
		float rect_area = (first_rect.height)*(first_rect.width);
		float img_area = (received_image.rows) * (received_image.cols);
		RCLCPP_DEBUG_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Valid area bounding_rect=(x=%d,y=%d,w=%d,h=%d) area_ratio=%.2f%%",
			first_rect.x,
			first_rect.y,
			first_rect.width,
			first_rect.height,
			(rect_area / img_area) * 100.0f);

		cv::Mat cropped_img;
		received_image(first_rect).copyTo(cropped_img);
		if (cropped_img.empty()) {
			RCLCPP_WARN(this->get_logger(), "Skipping occupancy grid because the cropped image is empty.");
			reject_region_update(map, "cropped_image_empty");
			return;
		}

		cv::Mat image_cleaned = cv::Mat::zeros(received_image.size(), CV_8UC1);
		cv::Mat black_image   = cv::Mat::zeros(received_image.size(), CV_8UC1);

		cv::Mat black_image2, image_cleaned2 = clean_image2(cropped_img, black_image2);
		cv::Mat cleaned_clone = image_cleaned2.clone();
		std::vector<std::vector<cv::Point>> cleaned_contours;
		cv::findContours(cleaned_clone, cleaned_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		RCLCPP_DEBUG_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Cleaned occupancy extracted_contours=%zu cleaned_free_pixels=%d black_pixels=%d",
			cleaned_contours.size(),
			cv::countNonZero(image_cleaned2),
			cv::countNonZero(black_image2));

		image_cleaned2.copyTo(image_cleaned(first_rect));
		black_image2.copyTo(black_image(first_rect));
		cv::flip(image_cleaned, image2save_clean_,0);

		end_process = getTime();
		occupancy_time = end_process - begin_process;

	///////////////////////// Decompose Image
		begin_process = getTime();

		try{
			Stable_ = inc_decomp_.decompose_image(image_cleaned, pixel_Tau, origin, map->info.resolution);
		}
		catch (const std::exception &ex) {
			RCLCPP_WARN(
				this->get_logger(),
				"Decomposition threw '%s'; keeping previous region state for this frame.",
				ex.what());
		}
		catch (...) {
			RCLCPP_WARN(
				this->get_logger(),
				"Decomposition threw an unknown exception; keeping previous region state for this frame.");
		}
		const char *rejection_reason =
			inc_decomp_.last_rejection_reason_.empty() ? "none" : inc_decomp_.last_rejection_reason_.c_str();

		end_process = getTime();
		decompose_time = end_process - begin_process;

	////////////Draw Image & publish
		begin_process = getTime();

		cv::flip(black_image, black_image,0);
		image2save_black_ = black_image.clone();
		grad = Stable_.draw_stable_contour() & ~black_image;
		image2save_Inc_ = grad.clone();

		cv_ptr_->encoding = sensor_msgs::image_encodings::RGB8;
		cv_ptr_->image = make_region_visualization(Stable_, black_image);

		end_process = getTime();
		drawPublish_time = end_process - begin_process;
		whole_time = end_process - begin_whole;

		++processed_maps_;
		cumulative_whole_time_ms_ += whole_time;
		cumulative_squared_whole_time_ms_ += whole_time * whole_time;
		const double avg_time = cumulative_whole_time_ms_ / processed_maps_;
		const double avg_sq_time = cumulative_squared_whole_time_ms_ / processed_maps_;
		const double std_time = std::sqrt(std::max(0.0, avg_sq_time - avg_time * avg_time));

		RCLCPP_INFO_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Processed map %ux%u @ %.3f m/pix: regions=%zu branch=%s total_ms=%.1f avg_ms=%.1f",
			map->info.width,
			map->info.height,
			map->info.resolution,
			Stable_.Region_contour.size(),
			inc_decomp_.last_branch_.c_str(),
			whole_time,
			avg_time);
		RCLCPP_DEBUG_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Timing occ_ms=%.1f decomp_ms=%.1f draw_ms=%.1f total_ms=%.1f avg_ms=%.1f std_ms=%.1f rejection_reason=%s",
			occupancy_time,
			decompose_time,
			drawPublish_time,
			whole_time,
			avg_time,
			std_time,
			rejection_reason);

		const bool accepted_update = publish_regions_with_validation(
			map,
			free_cells,
			occupied_cells,
			unknown_cells,
			expected_cells);
		if (accepted_update) {
			profiler_->record("dude_preprocess_ms", occupancy_time);
			profiler_->record("dude_decomposition_ms", decompose_time);
			profiler_->record("dude_draw_publish_ms", drawPublish_time);
			profiler_->record("dude_callback_total_ms", whole_time);
			maybeSaveProfilingSnapshot();
		}
		RCLCPP_INFO_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Published regions for map %ux%u: source_regions=%zu branch=%s",
			map->info.width,
			map->info.height,
			Stable_.Region_contour.size(),
			inc_decomp_.last_branch_.c_str());
	}

	void metronomeCallback()
	{
		publish_Image();
	}

	void chatCallback(const std_msgs::msg::String::SharedPtr chat_msg)
	{
		std::string saving_path = "src/incremental_dude_ros2/incremental_dude_ros2/maps/Topological_Segmentation/";
		std::filesystem::create_directories(saving_path);
		RCLCPP_INFO(
			this->get_logger(),
			"Saving comparison images with tag '%s' into %s",
			chat_msg->data.c_str(),
			saving_path.c_str());
		cv::Mat proxy, zero =cv::Mat::zeros(image2save_clean_.size(),CV_8U);
		(void)zero;
		cv::Mat Batch_segmentated = simple_segment(image2save_clean_);
		std::map<int,int> Batch_Inc_map = compare_images(Batch_segmentated, image2save_Inc_);

		Batch_segmentated.copyTo(proxy , ~image2save_black_);
		Batch_segmentated = proxy.clone();

		double min, max_batch, max_inc;
		cv::minMaxLoc(Batch_segmentated, &min, &max_batch);
		cv::minMaxLoc(image2save_Inc_, &min, &max_inc);
		(void)max_batch;
		(void)max_inc;

		cv::Mat destroyable_batch = Batch_segmentated.clone();
		std::vector<std::vector<cv::Point> > test_contour;
		cv::findContours(destroyable_batch, test_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		cv::Rect first_rect = cv::boundingRect(test_contour[0]);
		for(size_t i=1; i < test_contour.size(); i++){
			first_rect |= cv::boundingRect(test_contour[i]);
		}

		cv::Mat cropped_Batch, cropped_Inc;
		Batch_segmentated(first_rect).copyTo(cropped_Batch);
		image2save_Inc_(first_rect).copyTo(cropped_Inc);

		if ( true ){
			std::vector <cv::Vec3b> colormap = save_image_original_color(saving_path + chat_msg->data + "_Batch.png", cropped_Batch);
			save_decomposed_image_color(saving_path + chat_msg->data + "_Inc.png", cropped_Inc, colormap, Batch_Inc_map);
		}
		else{
			std::vector <cv::Vec3b> colormap = save_image_original_color(saving_path + chat_msg->data + "_Inc.png", image2save_Inc_ );
			save_decomposed_image_color(saving_path + chat_msg->data + "_Batch.png", Batch_segmentated, colormap, Batch_Inc_map);
		}
	}

	void publish_Image(){
		if (!cv_ptr_ || cv_ptr_->image.empty()) {
			return;
		}
		image_pub_.publish(cv_ptr_->toImageMsg());
	}

	void reset_outputs(const cv::Size& size)
	{
		Stable_.Region_contour.clear();
		Stable_.Region_centroid.clear();
		Stable_.image_size = size;
		image2save_clean_ = cv::Mat::zeros(size, CV_8UC1);
		image2save_black_ = cv::Mat::zeros(size, CV_8UC1);
		image2save_Inc_ = cv::Mat::zeros(size, CV_8UC1);
		if (cv_ptr_) {
			cv_ptr_->image = cv::Mat();
		}
	}

	incremental_dude_msgs::msg::Region2DArray build_raw_region_message(
		const nav_msgs::msg::OccupancyGrid::SharedPtr& map)
	{
		auto msg = incremental_dude_msgs::msg::Region2DArray();
		msg.header = map->header;

		std::vector<PublishedContour> published_contours;
		published_contours.reserve(Stable_.Region_contour.size());

		for(size_t i = 0; i < Stable_.Region_contour.size(); ++i){
			if (!is_publishable_contour(Stable_.Region_contour[i])) {
				continue;
			}
			PublishedContour contour_view;
			contour_view.id = static_cast<int>(published_contours.size());
			contour_view.contour = &Stable_.Region_contour[i];
			published_contours.push_back(contour_view);
		}

		std::vector<std::vector<int> > adjacency(published_contours.size());
		for(size_t i = 0; i < published_contours.size(); ++i){
			for(size_t j = i + 1; j < published_contours.size(); ++j){
				cv::Point centroid;
				int connected = 0;
				inc_decomp_.are_contours_connected(
					*published_contours[i].contour,
					*published_contours[j].contour,
					centroid,
					connected);
				(void)centroid;
				if (connected > 0){
					adjacency[i].push_back(published_contours[j].id);
					adjacency[j].push_back(published_contours[i].id);
				}
			}
		}

		CoordinateConverter converter(
			map->info.resolution,
			map->info.width,
			map->info.height,
			map->info.origin.position.x,
			map->info.origin.position.y);

		for(size_t i = 0; i < published_contours.size(); ++i){
			incremental_dude_msgs::msg::Region2D region_msg;
			region_msg.header = map->header;
			region_msg.id = published_contours[i].id;

			std::vector<cv::Point> polygon_contour = *published_contours[i].contour;
			ensure_counter_clockwise(polygon_contour);

			cv::Moments moments = cv::moments(polygon_contour, true);
			cv::Point centroid;
			if (moments.m00 != 0.0) {
				centroid = cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
			} else {
				centroid = polygon_contour.front();
			}

			region_msg.centroid = RegionROSConverter::pointToPose2D(centroid, converter);
			region_msg.polygon = RegionROSConverter::contourToPolygon(polygon_contour, converter);

			std::vector<cv::Point> hull;
			cv::convexHull(polygon_contour, hull);
			ensure_counter_clockwise(hull);
			region_msg.convex_hull = RegionROSConverter::contourToPolygon(hull, converter);

			region_msg.area =
				std::abs(cv::contourArea(polygon_contour)) *
				map->info.resolution * map->info.resolution;
			region_msg.adjacent_ids = adjacency[i];

			msg.regions.push_back(region_msg);
		}

		RCLCPP_DEBUG_THROTTLE(
			this->get_logger(),
			*this->get_clock(),
			5000,
			"Publishing regions source_regions=%zu valid_regions=%zu raw_published=%zu selected_branch=%s fallback_or_rejection=%s accepted_for_publish=%s",
			Stable_.Region_contour.size(),
			published_contours.size(),
			msg.regions.size(),
			inc_decomp_.last_branch_.c_str(),
			inc_decomp_.last_rejection_reason_.empty() ? "none" : inc_decomp_.last_rejection_reason_.c_str(),
			published_contours.empty() && !Stable_.Region_contour.empty() ? "false" : "true");

		return msg;
	}

	bool publish_regions_with_validation(
		const nav_msgs::msg::OccupancyGrid::SharedPtr& map,
		const size_t free_cells,
		const size_t occupied_cells,
		const size_t unknown_cells,
		const size_t expected_cells)
	{
		(void)occupied_cells;
		auto msg = build_raw_region_message(map);
		const RegionUpdateValidation validation =
			validate_region_update(msg, free_cells, unknown_cells, expected_cells);

		if (!validation.accepted) {
			reject_region_update(map, validation.reason);
			return false;
		}

		if (!region_tracker_enable_) {
			region_pub_->publish(msg);
			remember_valid_region_message(msg, validation);
			RCLCPP_INFO(
				this->get_logger(),
				"Accepted decomposition update regions=%zu total_area=%.3f unknown_ratio=%.3f free_ratio=%.3f branch=%s",
				validation.region_count,
				validation.total_area,
				validation.unknown_ratio,
				validation.free_ratio,
				inc_decomp_.last_branch_.c_str());
			return true;
		}

		const double tracking_start_ms = getTime();
		const auto tracked = region_tracker_.canonicalize(msg);
		profiler_->record("region_tracking_ms", getTime() - tracking_start_ms);
		if (region_tracker_publish_debug_) {
			RCLCPP_INFO_THROTTLE(
				this->get_logger(),
				*this->get_clock(),
				1000,
				"Region tracker published=%zu matched=%zu new=%zu active_tracks=%zu removed_tracks=%zu",
				tracked.message.regions.size(),
				tracked.stats.matched_regions,
				tracked.stats.new_regions,
				tracked.stats.active_tracks,
				tracked.stats.removed_tracks);
		}
		region_pub_->publish(tracked.message);
		remember_valid_region_message(tracked.message, validation);
		RCLCPP_INFO(
			this->get_logger(),
			"Accepted decomposition update regions=%zu tracked_regions=%zu total_area=%.3f unknown_ratio=%.3f free_ratio=%.3f branch=%s",
			validation.region_count,
			tracked.message.regions.size(),
			validation.total_area,
			validation.unknown_ratio,
			validation.free_ratio,
			inc_decomp_.last_branch_.c_str());
		return true;
	}

	RegionUpdateValidation validate_region_update(
		const incremental_dude_msgs::msg::Region2DArray& msg,
		const size_t free_cells,
		const size_t unknown_cells,
		const size_t expected_cells) const
	{
		RegionUpdateValidation validation;
		validation.region_count = msg.regions.size();
		for (const auto& region : msg.regions) {
			if (std::isfinite(region.area) && region.area > 0.0f) {
				validation.total_area += static_cast<double>(region.area);
			}
		}
		if (expected_cells > 0) {
			validation.unknown_ratio =
				static_cast<double>(unknown_cells) / static_cast<double>(expected_cells);
			validation.free_ratio =
				static_cast<double>(free_cells) / static_cast<double>(expected_cells);
		}

		const int min_regions = std::max(0, min_valid_region_count_);
		if (!has_last_valid_region_msg_) {
			if (validation.region_count < static_cast<size_t>(min_regions)) {
				validation.accepted = false;
				validation.reason = "initial_update_below_min_region_count";
			}
			return validation;
		}

		std::vector<std::string> reasons;
		if (validation.region_count == 0 && last_valid_region_count_ > 0) {
			reasons.push_back("zero_regions_after_valid_update");
		} else if (validation.region_count < static_cast<size_t>(min_regions)) {
			reasons.push_back("below_min_valid_region_count");
		}

		if (validation.unknown_ratio > max_unknown_ratio_for_update_) {
			reasons.push_back("unknown_ratio_too_high");
		}

		if (last_valid_region_area_ > 0.0) {
			const double area_ratio = validation.total_area / last_valid_region_area_;
			const double min_area_ratio =
				std::max(0.0, 1.0 - std::max(0.0, max_region_area_drop_ratio_));
			if (area_ratio < min_area_ratio) {
				reasons.push_back("region_area_drop_too_large");
			}
		}

		if (last_valid_region_count_ > 0) {
			const double count_ratio =
				static_cast<double>(validation.region_count) /
				static_cast<double>(last_valid_region_count_);
			const double min_count_ratio =
				std::max(0.0, 1.0 - std::max(0.0, max_region_count_drop_ratio_));
			if (count_ratio < min_count_ratio) {
				reasons.push_back("region_count_drop_too_large");
			}
		}

		if (!reasons.empty()) {
			validation.accepted = false;
			std::ostringstream stream;
			for (size_t i = 0; i < reasons.size(); ++i) {
				if (i > 0) {
					stream << ";";
				}
				stream << reasons[i];
			}
			stream
				<< " current_regions=" << validation.region_count
				<< " previous_regions=" << last_valid_region_count_
				<< " current_area=" << validation.total_area
				<< " previous_area=" << last_valid_region_area_
				<< " unknown_ratio=" << validation.unknown_ratio
				<< " free_ratio=" << validation.free_ratio
				<< " branch=" << inc_decomp_.last_branch_;
			validation.reason = stream.str();
		}
		return validation;
	}

	void remember_valid_region_message(
		const incremental_dude_msgs::msg::Region2DArray& msg,
		const RegionUpdateValidation& validation)
	{
		last_valid_region_msg_ = msg;
		last_valid_region_count_ = validation.region_count;
		last_valid_region_area_ = validation.total_area;
		last_valid_stable_ = Stable_;
		has_last_valid_region_msg_ = true;
		rejected_region_updates_ = 0;
	}

	void reject_region_update(
		const nav_msgs::msg::OccupancyGrid::SharedPtr& map,
		const std::string& reason)
	{
		++rejected_region_updates_;
		RCLCPP_WARN(
			this->get_logger(),
			"Rejected suspicious decomposition update reason=%s consecutive_rejections=%zu have_previous=%s preserve_previous=%s publish_stale=%s",
			reason.c_str(),
			rejected_region_updates_,
			has_last_valid_region_msg_ ? "true" : "false",
			preserve_previous_on_invalid_update_ ? "true" : "false",
			publish_stale_regions_on_reject_ ? "true" : "false");

		if (preserve_previous_on_invalid_update_ && has_last_valid_region_msg_) {
			Stable_ = last_valid_stable_;
			inc_decomp_.Stable = last_valid_stable_;
			if (publish_stale_regions_on_reject_) {
				auto stale_msg = last_valid_region_msg_;
				stale_msg.header = map->header;
				for (auto& region : stale_msg.regions) {
					region.header = map->header;
				}
				region_pub_->publish(stale_msg);
				RCLCPP_WARN(
					this->get_logger(),
					"Published stale regions from last valid decomposition regions=%zu reason=%s",
					stale_msg.regions.size(),
					reason.c_str());
			} else {
				RCLCPP_WARN(
					this->get_logger(),
					"Preserved previous valid regions without publishing stale update reason=%s",
					reason.c_str());
			}
			return;
		}

		auto empty_msg = incremental_dude_msgs::msg::Region2DArray();
		empty_msg.header = map->header;
		region_pub_->publish(empty_msg);
		RCLCPP_WARN(
			this->get_logger(),
			"Published empty regions because no previous valid decomposition is available reason=%s",
			reason.c_str());
	}

	static void ensure_counter_clockwise(std::vector<cv::Point>& contour)
	{
		if (contour.size() < 3) {
			return;
		}
		if (cv::contourArea(contour, true) < 0.0) {
			std::reverse(contour.begin(), contour.end());
		}
	}

	static bool is_publishable_contour(const std::vector<cv::Point>& contour)
	{
		if (contour.size() < 3) {
			return false;
		}
		const double area = std::abs(cv::contourArea(contour));
		if (!std::isfinite(area) || area <= 1.0) {
			return false;
		}
		const cv::Rect bounds = cv::boundingRect(contour);
		return bounds.width > 0 && bounds.height > 0;
	}

	static cv::Vec3b region_id_to_rgb(const int region_id)
	{
		if (region_id <= 0) {
			return cv::Vec3b(0, 0, 0);
		}

		constexpr double golden_ratio_conjugate = 0.6180339887498949;
		const double hue = std::fmod(region_id * golden_ratio_conjugate, 1.0) * 6.0;
		const double saturation = 0.82;
		const double value = 0.95;
		const int sector = static_cast<int>(std::floor(hue));
		const double fraction = hue - sector;
		const double p = value * (1.0 - saturation);
		const double q = value * (1.0 - saturation * fraction);
		const double t = value * (1.0 - saturation * (1.0 - fraction));

		double red = 0.0;
		double green = 0.0;
		double blue = 0.0;
		switch (sector % 6) {
		case 0:
			red = value;
			green = t;
			blue = p;
			break;
		case 1:
			red = q;
			green = value;
			blue = p;
			break;
		case 2:
			red = p;
			green = value;
			blue = t;
			break;
		case 3:
			red = p;
			green = q;
			blue = value;
			break;
		case 4:
			red = t;
			green = p;
			blue = value;
			break;
		default:
			red = value;
			green = p;
			blue = q;
			break;
		}

		return cv::Vec3b(
			static_cast<unsigned char>(std::round(red * 255.0)),
			static_cast<unsigned char>(std::round(green * 255.0)),
			static_cast<unsigned char>(std::round(blue * 255.0)));
	}

	static cv::Mat make_region_visualization(
		const Stable_graph& stable_graph,
		const cv::Mat& flipped_obstacle_mask)
	{
		cv::Mat visualization = cv::Mat::zeros(stable_graph.image_size, CV_8UC3);
		for (size_t i = 0; i < stable_graph.Region_contour.size(); ++i) {
			const int region_id = static_cast<int>(i) + 1;
			const cv::Vec3b rgb = region_id_to_rgb(region_id);
			cv::drawContours(
				visualization,
				stable_graph.Region_contour,
				static_cast<int>(i),
				cv::Scalar(rgb[0], rgb[1], rgb[2]),
				-1,
				8);
		}

		cv::flip(visualization, visualization, 0);
		if (!flipped_obstacle_mask.empty() &&
			flipped_obstacle_mask.size() == visualization.size()) {
			visualization.setTo(cv::Vec3b(0, 0, 0), flipped_obstacle_mask);
		}
		return visualization;
	}

	cv::Mat clean_image2(cv::Mat Occ_Image, cv::Mat &black_image){
		cv::Mat open_space = Occ_Image<10;
		black_image = Occ_Image>90 & Occ_Image<=100;
		cv::Mat Median_Image, out_image, temp_image;
		int filter_size=2;

		cv::boxFilter(black_image, temp_image, -1, cv::Size(filter_size, filter_size), cv::Point(-1,-1), false, cv::BORDER_DEFAULT );
		black_image = temp_image > filter_size*filter_size/2;
		cv::dilate(black_image, black_image, cv::Mat(), cv::Point(-1,-1), 4, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue() );

		filter_size=10;
		cv::boxFilter(open_space, temp_image, -1, cv::Size(filter_size, filter_size), cv::Point(-1,-1), false, cv::BORDER_DEFAULT );
		Median_Image = temp_image > filter_size*filter_size/2;
		Median_Image = Median_Image | open_space ;
		cv::dilate(Median_Image, Median_Image,cv::Mat());

		out_image = Median_Image & ~black_image;

		return out_image;
	}

	cv::Rect find_image_bounding_Rect(cv::Mat Occ_Image){
		cv::Mat valid_image = Occ_Image < 101;
		std::vector<std::vector<cv::Point> > test_contour;
		cv::findContours(valid_image, test_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		if (test_contour.empty()) {
			return cv::Rect(0, 0, Occ_Image.cols, Occ_Image.rows);
		}

		cv::Rect first_rect = cv::boundingRect(test_contour[0]);
		for(size_t i=1; i < test_contour.size(); i++){
			first_rect |= cv::boundingRect(test_contour[i]);
		}
		return first_rect;
	}

	typedef std::map <std::vector<int>, std::vector <cv::Point> > match2points;
	typedef std::map <int, std::vector <cv::Point> > tag2points;
	typedef std::map <int, tag2points> tag2tagMapper;

	void save_decomposed_image_color(std::string path, cv::Mat image_in, std::vector <cv::Vec3b> colormap, std::map<int,int> original_map){
		double min, max;
		std::vector <cv::Vec3b> color_vector;
		cv::Vec3b black(208, 208, 208);
		color_vector.push_back(black);

		std::map<int,int>::iterator map_iter;

		cv::minMaxLoc(image_in, &min,&max);
		color_vector.resize(max);

		for(int i=1;i<= max; i++){
			map_iter = original_map.find(i);
			if (map_iter != original_map.end()){
				int index_in_original = map_iter->second;
				color_vector[i]=colormap[index_in_original];
			}
			else{
				cv::Vec3b color(rand() % 255,rand() % 255,rand() % 255);
				color_vector[i] = color;
			}
		}
		cv::Mat image_float = cv::Mat::zeros(image_in.size(), CV_8UC3);
		for(int i=0; i < image_in.rows; i++){
			for(int j=0;j< image_in.cols; j++){
				int color_index = image_in.at<uchar>(i,j);
				image_float.at<cv::Vec3b>(i,j) = color_vector[color_index];
			}
		}
		cv::imwrite(path , image_float);
	}

	std::vector <cv::Vec3b> save_image_original_color(std::string path, cv::Mat image_in){
		double min, max;
		std::vector <cv::Vec3b> color_vector;
		cv::Vec3b black(208, 208, 208);
		color_vector.push_back(black);

		cv::minMaxLoc(image_in, &min,&max);

		for(int i=0;i<= max; i++){
			cv::Vec3b color(rand() % 255,rand() % 255,rand() % 255);
			color_vector.push_back(color);
		}
		cv::Mat image_float = cv::Mat::zeros(image_in.size(), CV_8UC3);
		for(int i=0; i < image_in.rows; i++){
			for(int j=0;j< image_in.cols; j++){
				int color_index = image_in.at<uchar>(i,j);
				image_float.at<cv::Vec3b>(i,j) = color_vector[color_index];
			}
		}
		cv::imwrite(path , image_float);

		return color_vector;
	}

	cv::Mat simple_segment(cv::Mat image_in){
		Incremental_Decomposer inc_decomp;
		Stable_graph Stable;
		cv::Point2f origin(0,0);
		float resolution = 0.05;

		cv::Mat pre_decompose = image_in.clone();
		cv::Mat pre_decompose_BW = pre_decompose > 250;

		Stable = inc_decomp.decompose_image(pre_decompose_BW, Decomp_threshold_/resolution, origin , resolution);

		cv::Mat Drawing = cv::Mat::zeros(image_in.size(), CV_8UC1);
		for(int i = 0; i < static_cast<int>(Stable.Region_contour.size());i++){
			drawContours(Drawing, Stable.Region_contour, i, i+1, -1, 8);
		}
		RCLCPP_DEBUG(
			this->get_logger(),
			"Batch segmentation produced %zu regions",
			Stable.Region_contour.size());

		return Drawing;
	}

	std::map<int,int> compare_images(cv::Mat GT_segmentation_in, cv::Mat DuDe_segmentation_in){

		std::map<int,int> segmented2GT_tags;

		cv::Mat GT_segmentation   = cv::Mat::zeros(GT_segmentation_in.size(),CV_8UC1);
		cv::Mat DuDe_segmentation = cv::Mat::zeros(GT_segmentation_in.size(),CV_8UC1);

		GT_segmentation_in  .convertTo(GT_segmentation, CV_8UC1);
		DuDe_segmentation_in.convertTo(DuDe_segmentation, CV_8UC1);
		tag2tagMapper gt_tag2mapper,DuDe_tag2mapper;

		match2points links2points;
		tag2points GT_points, DuDe_points;

		for(int x=0; x < GT_segmentation.size().width; x++){
			for(int y=0; y < GT_segmentation.size().height; y++){
				cv::Point current_pixel(x,y);
				std::vector<int> match;

				int tag_GT   = GT_segmentation.at<uchar>(current_pixel);
				int tag_DuDe  = DuDe_segmentation.at<uchar>(current_pixel);

				if(tag_DuDe>0 && tag_GT>0 ){
					gt_tag2mapper  [tag_GT][tag_DuDe].push_back(current_pixel);
					DuDe_tag2mapper[tag_DuDe][tag_GT].push_back(current_pixel);
					match.push_back(tag_DuDe);
					match.push_back(tag_GT);
					links2points[match].push_back(current_pixel);
					GT_points[tag_GT].push_back(current_pixel);
					DuDe_points[tag_DuDe].push_back(current_pixel);
				}
			}
		}

		std::map <std::vector<int>, float > link2relation;
		std::map<int,int> DuDe_Union_Match;
		int current_DuDe_Tag=0;
		int current_GT_max = -1;
		float current_GT_relation = 0;

		for( tag2tagMapper::iterator it = gt_tag2mapper.begin(); it!= gt_tag2mapper.end(); it++ ){
			int tag_GT = it->first;
			int big_DuDe_tag = -1;
			int big_match = -1;

			for(tag2points::iterator inside_it = gt_tag2mapper[tag_GT].begin(); inside_it != gt_tag2mapper[tag_GT].end(); inside_it++ ){
				int tag_DuDe = inside_it->first;
				int current_match = gt_tag2mapper[tag_GT][tag_DuDe].size();

				if( current_match > big_match ){
					big_match = current_match;
					big_DuDe_tag = tag_DuDe;
				}
			}
			segmented2GT_tags[big_DuDe_tag] = tag_GT;
		}

		for( tag2tagMapper::iterator it = DuDe_tag2mapper.begin(); it!= DuDe_tag2mapper.end(); it++ ){
			int tag_DuDe = it->first;

			int union_match = 0;
			for(tag2points::iterator inside_it = DuDe_tag2mapper[tag_DuDe].begin(); inside_it != DuDe_tag2mapper[tag_DuDe].end(); inside_it++ ){
				int tag_GT = inside_it->first;
				int current_match = DuDe_tag2mapper[tag_DuDe][tag_GT].size();

				if( current_match > current_GT_relation ){
					current_GT_relation = current_match;
					current_GT_max = tag_GT;
				}
				union_match += current_match;
			}
			current_DuDe_Tag = tag_DuDe;
			if(static_cast<float>(current_GT_relation)/union_match > 0.5){
				DuDe_Union_Match[current_DuDe_Tag] = current_GT_max;
			}
		}

		for(std::map<int,int>::iterator it = DuDe_Union_Match.begin(); it != DuDe_Union_Match.end(); it++){
			segmented2GT_tags[it->first] = it->second;
		}

		return segmented2GT_tags;
	}
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);

	auto node = std::make_shared<ROS_handler>();
	node->init_publishers();
	rclcpp::on_shutdown([node]() {
		node->saveProfiling();
	});

	rclcpp::spin(node);
	node->saveProfiling();
	rclcpp::shutdown();

	return 0;
}
