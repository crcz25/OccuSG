#include "inc_decomp.hpp"
#include "dude_ros_logging.hpp"

#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr double kMinStableContourAreaPixels = 1.0;

struct WrapperRunResult {
	bool ok{false};
	bool child_signaled{false};
	int child_status{0};
	cv::Rect resize_rect;
	std::vector<std::vector<cv::Point>> decomposed_contours;
	std::vector<cv::Point> contours_centroid;
	size_t input_contours{0};
	size_t valid_polygons{0};
	size_t rejected_polygons{0};
	size_t cuts_after_resolution{0};
};

bool writeAll(const int fd, const void *data, const size_t size) {
	const auto *bytes = static_cast<const uint8_t *>(data);
	size_t written = 0;
	while (written < size) {
		const ssize_t rc = ::write(fd, bytes + written, size - written);
		if (rc < 0) {
			if (errno == EINTR) {
				continue;
			}
			return false;
		}
		if (rc == 0) {
			return false;
		}
		written += static_cast<size_t>(rc);
	}
	return true;
}

bool readAll(const int fd, void *data, const size_t size) {
	auto *bytes = static_cast<uint8_t *>(data);
	size_t read_bytes = 0;
	while (read_bytes < size) {
		const ssize_t rc = ::read(fd, bytes + read_bytes, size - read_bytes);
		if (rc < 0) {
			if (errno == EINTR) {
				continue;
			}
			return false;
		}
		if (rc == 0) {
			return false;
		}
		read_bytes += static_cast<size_t>(rc);
	}
	return true;
}

template <typename T>
bool writePod(const int fd, const T &value) {
	return writeAll(fd, &value, sizeof(T));
}

template <typename T>
bool readPod(const int fd, T &value) {
	return readAll(fd, &value, sizeof(T));
}

bool writePoint(const int fd, const cv::Point &point) {
	const int32_t x = point.x;
	const int32_t y = point.y;
	return writePod(fd, x) && writePod(fd, y);
}

bool readPoint(const int fd, cv::Point &point) {
	int32_t x = 0;
	int32_t y = 0;
	if (!readPod(fd, x) || !readPod(fd, y)) {
		return false;
	}
	point = cv::Point(x, y);
	return true;
}

bool writeWrapperResult(const int fd, const DuDe_OpenCV_wrapper &wrapper, const cv::Rect &rect) {
	const uint8_t success = 1;
	if (!writePod(fd, success)) {
		return false;
	}
	const int32_t rect_values[4] = {rect.x, rect.y, rect.width, rect.height};
	if (!writeAll(fd, rect_values, sizeof(rect_values))) {
		return false;
	}
	const uint64_t diagnostics[4] = {
		static_cast<uint64_t>(wrapper.last_input_contours_),
		static_cast<uint64_t>(wrapper.last_valid_polygons_),
		static_cast<uint64_t>(wrapper.last_rejected_polygons_),
		static_cast<uint64_t>(wrapper.last_cuts_after_resolution_)};
	if (!writeAll(fd, diagnostics, sizeof(diagnostics))) {
		return false;
	}
	const uint64_t contour_count = wrapper.Decomposed_contours.size();
	if (!writePod(fd, contour_count)) {
		return false;
	}
	for (const auto &contour : wrapper.Decomposed_contours) {
		const uint64_t point_count = contour.size();
		if (!writePod(fd, point_count)) {
			return false;
		}
		for (const auto &point : contour) {
			if (!writePoint(fd, point)) {
				return false;
			}
		}
	}
	const uint64_t centroid_count = wrapper.contours_centroid.size();
	if (!writePod(fd, centroid_count)) {
		return false;
	}
	for (const auto &centroid : wrapper.contours_centroid) {
		if (!writePoint(fd, centroid)) {
			return false;
		}
	}
	return true;
}

bool readWrapperResult(const int fd, WrapperRunResult &result) {
	uint8_t success = 0;
	if (!readPod(fd, success) || success != 1) {
		return false;
	}
	int32_t rect_values[4] = {};
	if (!readAll(fd, rect_values, sizeof(rect_values))) {
		return false;
	}
	result.resize_rect = cv::Rect(rect_values[0], rect_values[1], rect_values[2], rect_values[3]);
	uint64_t diagnostics[4] = {};
	if (!readAll(fd, diagnostics, sizeof(diagnostics))) {
		return false;
	}
	result.input_contours = static_cast<size_t>(diagnostics[0]);
	result.valid_polygons = static_cast<size_t>(diagnostics[1]);
	result.rejected_polygons = static_cast<size_t>(diagnostics[2]);
	result.cuts_after_resolution = static_cast<size_t>(diagnostics[3]);
	uint64_t contour_count = 0;
	if (!readPod(fd, contour_count)) {
		return false;
	}
	result.decomposed_contours.clear();
	result.decomposed_contours.reserve(static_cast<size_t>(contour_count));
	for (uint64_t i = 0; i < contour_count; ++i) {
		uint64_t point_count = 0;
		if (!readPod(fd, point_count)) {
			return false;
		}
		std::vector<cv::Point> contour;
		contour.reserve(static_cast<size_t>(point_count));
		for (uint64_t j = 0; j < point_count; ++j) {
			cv::Point point;
			if (!readPoint(fd, point)) {
				return false;
			}
			contour.push_back(point);
		}
		result.decomposed_contours.push_back(std::move(contour));
	}
	uint64_t centroid_count = 0;
	if (!readPod(fd, centroid_count)) {
		return false;
	}
	result.contours_centroid.clear();
	result.contours_centroid.reserve(static_cast<size_t>(centroid_count));
	for (uint64_t i = 0; i < centroid_count; ++i) {
		cv::Point centroid;
		if (!readPoint(fd, centroid)) {
			return false;
		}
		result.contours_centroid.push_back(centroid);
	}
	result.ok = true;
	return true;
}

WrapperRunResult runWrapperInChild(const cv::Mat &image, const float pixel_tau, const char *context) {
	WrapperRunResult result;
	int pipe_fds[2] = {-1, -1};
	if (::pipe(pipe_fds) != 0) {
		DUDE_ROS_WARNF(
			"decompose_image wrapper_child context=%s reason=pipe_failed error=%s",
			context,
			std::strerror(errno));
		return result;
	}

	const pid_t pid = ::fork();
	if (pid < 0) {
		DUDE_ROS_WARNF(
			"decompose_image wrapper_child context=%s reason=fork_failed error=%s",
			context,
			std::strerror(errno));
		::close(pipe_fds[0]);
		::close(pipe_fds[1]);
		return result;
	}

	if (pid == 0) {
		::close(pipe_fds[0]);
		DuDe_OpenCV_wrapper wrapper;
		wrapper.set_pixel_Tau(pixel_tau);
		const cv::Rect rect = wrapper.Decomposer(image);
		const bool wrote = writeWrapperResult(pipe_fds[1], wrapper, rect);
		::close(pipe_fds[1]);
		::_exit(wrote ? 0 : 2);
	}

	::close(pipe_fds[1]);
	const bool read_ok = readWrapperResult(pipe_fds[0], result);
	::close(pipe_fds[0]);

	int status = 0;
	while (::waitpid(pid, &status, 0) < 0) {
		if (errno == EINTR) {
			continue;
		}
		DUDE_ROS_WARNF(
			"decompose_image wrapper_child context=%s reason=wait_failed error=%s",
			context,
			std::strerror(errno));
		result.ok = false;
		return result;
	}
	result.child_status = status;
	result.child_signaled = WIFSIGNALED(status);
	if (!read_ok || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		result.ok = false;
		if (WIFSIGNALED(status)) {
			DUDE_ROS_WARNF(
				"decompose_image wrapper_child context=%s rejected=true signal=%d",
				context,
				WTERMSIG(status));
		} else {
			DUDE_ROS_WARNF(
				"decompose_image wrapper_child context=%s rejected=true exit_status=%d read_ok=%s",
				context,
				WIFEXITED(status) ? WEXITSTATUS(status) : -1,
				read_ok ? "true" : "false");
		}
		return result;
	}

	DUDE_ROS_DEBUGF(
		"decompose_image wrapper_child context=%s ok=true input_contours=%zu polygons=%zu rejected_polygons=%zu cuts_after=%zu raw_decomposed_regions=%zu",
		context,
		result.input_contours,
		result.valid_polygons,
		result.rejected_polygons,
		result.cuts_after_resolution,
		result.decomposed_contours.size());
	return result;
}

bool isValidStableContour(const std::vector<cv::Point> &contour,
                          const double min_area = kMinStableContourAreaPixels) {
	if (contour.size() < 3) {
		return false;
	}
	const double area = std::abs(cv::contourArea(contour));
	if (!std::isfinite(area) || area <= min_area) {
		return false;
	}
	const cv::Rect bounds = cv::boundingRect(contour);
	return bounds.width > 0 && bounds.height > 0;
}

cv::Point contourCentroidOrFirst(const std::vector<cv::Point> &contour) {
	const cv::Moments moments = cv::moments(contour, true);
	if (moments.m00 != 0.0 && std::isfinite(moments.m00)) {
		return cv::Point(
			static_cast<int>(moments.m10 / moments.m00),
			static_cast<int>(moments.m01 / moments.m00));
	}
	return contour.empty() ? cv::Point(0, 0) : contour.front();
}

std::vector<std::vector<cv::Point>>
filterContoursByArea(const std::vector<std::vector<cv::Point>> &contours,
                     const double min_area) {
	std::vector<std::vector<cv::Point>> filtered;
	filtered.reserve(contours.size());
	for (const auto &contour : contours) {
		if (isValidStableContour(contour, min_area)) {
			filtered.push_back(contour);
		}
	}
	return filtered;
}

cv::Mat contoursToMask(const cv::Size &size,
                       const std::vector<std::vector<cv::Point>> &contours) {
	cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
	if (!contours.empty()) {
		drawContours(mask, contours, -1, 255, -1, 8);
	}
	return mask;
}

double computeContourOverlapRatio(
	const std::vector<std::vector<cv::Point>> &contours,
	const cv::Mat &reference_mask) {
	if (contours.empty() || reference_mask.empty()) {
		return 0.0;
	}
	const cv::Mat contour_mask = contoursToMask(reference_mask.size(), contours);
	const int contour_pixels = cv::countNonZero(contour_mask);
	if (contour_pixels == 0) {
		return 0.0;
	}
	cv::Mat overlap;
	cv::bitwise_and(contour_mask, reference_mask, overlap);
	return static_cast<double>(cv::countNonZero(overlap)) /
	       static_cast<double>(contour_pixels);
}

std::string summarizeContours(const std::vector<std::vector<cv::Point>> &contours) {
	std::ostringstream stream;
	stream << "count=" << contours.size();
	if (!contours.empty()) {
		double total_area = 0.0;
		double max_area = 0.0;
		for (const auto &contour : contours) {
			const double area = std::abs(cv::contourArea(contour));
			total_area += area;
			max_area = std::max(max_area, area);
		}
		stream << " total_area=" << total_area << " max_area=" << max_area;
	}
	return stream.str();
}

void applyCorrection(Stable_graph &graph, const cv::Point &correction) {
	for (size_t i = 0; i < graph.Region_contour.size(); ++i) {
		if (i < graph.Region_centroid.size()) {
			graph.Region_centroid[i] += correction;
		}
		for (auto &point : graph.Region_contour[i]) {
			point += correction;
		}
	}
}

Stable_graph buildGraphFromContours(
	const std::vector<std::vector<cv::Point>> &contours,
	const std::vector<cv::Point> &centroids,
	const cv::Size &image_size) {
	Stable_graph graph;
	graph.image_size = image_size;
	graph.Region_contour.reserve(contours.size());
	graph.Region_centroid.reserve(contours.size());
	for (size_t i = 0; i < contours.size(); ++i) {
		if (!isValidStableContour(contours[i])) {
			continue;
		}
		graph.Region_contour.push_back(contours[i]);
		if (i < centroids.size()) {
			graph.Region_centroid.push_back(centroids[i]);
		} else {
			graph.Region_centroid.push_back(contourCentroidOrFirst(contours[i]));
		}
	}
	return graph;
}

void sanitizeStableGraph(Stable_graph &graph,
                         const cv::Size &image_size,
                         const char *context) {
	std::vector<std::vector<cv::Point>> valid_contours;
	std::vector<cv::Point> valid_centroids;
	valid_contours.reserve(graph.Region_contour.size());
	valid_centroids.reserve(graph.Region_contour.size());
	size_t rejected = 0;
	for (size_t i = 0; i < graph.Region_contour.size(); ++i) {
		const auto &contour = graph.Region_contour[i];
		if (!isValidStableContour(contour)) {
			++rejected;
			continue;
		}
		valid_contours.push_back(contour);
		if (i < graph.Region_centroid.size()) {
			valid_centroids.push_back(graph.Region_centroid[i]);
		} else {
			valid_centroids.push_back(contourCentroidOrFirst(contour));
			++rejected;
		}
	}
	if (rejected > 0 || graph.Region_centroid.size() != graph.Region_contour.size()) {
		DUDE_ROS_WARNF(
			"decompose_image sanitized_graph context=%s input_regions=%zu input_centroids=%zu rejected_or_repaired=%zu output_regions=%zu",
			context,
			graph.Region_contour.size(),
			graph.Region_centroid.size(),
			rejected,
			valid_contours.size());
	}
	graph.Region_contour = std::move(valid_contours);
	graph.Region_centroid = std::move(valid_centroids);
	graph.image_size = image_size;
}

}  // namespace

cv::Mat Stable_graph::draw_stable_contour() const{
	cv::Mat Drawing = cv::Mat::zeros(image_size.height, image_size.width, CV_8UC1);	
	for(int i = 0; i < static_cast<int>(Region_contour.size());i++){
		drawContours(Drawing, Region_contour, i, i+1, -1, 8);
	}
	cv::flip(Drawing,Drawing,0);/*
	for(int i = 0; i < Region_centroid.size();i++){
		stringstream mix;      mix<<i;				std::string text = mix.str();
		putText(Drawing, text, cv::Point(Region_centroid[i].x, image_size.height - Region_centroid[i].y ), 
											cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Region_centroid.size()+1, 1, 8);
	}	*/
	return Drawing;
}




Incremental_Decomposer::Incremental_Decomposer(){
	Decomp_threshold_ = 2.5;
	resolution=0.05; //default;
	safety_distance = 0.5;//default
	first_time = true;
	current_origin_ = cv::Point2f(0,0);
	last_branch_.clear();
	last_rejection_reason_.clear();
}

Incremental_Decomposer::~Incremental_Decomposer(){
}

/////////////////////////////////
///// MAIN FUNCTION
const Stable_graph& Incremental_Decomposer::decompose_image(cv::Mat image_cleaned, float pixel_Tau, cv::Point2f origin, float resolution_in){
	
	new_origin_ = origin;// because it is first time
	resolution = resolution_in;
	last_branch_ = "start";
	last_rejection_reason_.clear();
	Stable.image_size = image_cleaned.size();

	if (image_cleaned.empty() || image_cleaned.rows == 0 || image_cleaned.cols == 0) {
		Stable.Region_contour.clear();
		Stable.Region_centroid.clear();
		previous_rect = cv::Rect();
		first_time = true;
		current_origin_ = new_origin_;
		last_branch_ = "reject_empty_image";
		last_rejection_reason_ = "image_cleaned_empty";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s image=%dx%d",
			last_branch_.c_str(),
			last_rejection_reason_.c_str(),
			image_cleaned.cols,
			image_cleaned.rows);
		return Stable;
	}

	DUDE_ROS_DEBUGF(
		"decompose_image branch=start image=%dx%d resolution=%.3f pixel_tau=%.3f first_time=%s previous_regions=%zu previous_rect=(%d,%d,%d,%d) current_origin=(%.3f,%.3f) new_origin=(%.3f,%.3f)",
		image_cleaned.cols,
		image_cleaned.rows,
		resolution,
		pixel_Tau,
		first_time ? "true" : "false",
		Stable.Region_contour.size(),
		previous_rect.x,
		previous_rect.y,
		previous_rect.width,
		previous_rect.height,
		current_origin_.x,
		current_origin_.y,
		new_origin_.x,
		new_origin_.y);
	
	clock_t begin_process, end_process;
	float elapsed_secs_process;
	(void)end_process;
	(void)elapsed_secs_process;
	cv::Rect resize_rect;
	const int gap = std::max(1, static_cast<int>(safety_distance / resolution));
	const double min_contour_area = static_cast<double>(gap * gap);
	const cv::Mat free_space_mask = image_cleaned > 0;
	cv::Mat full_clone = image_cleaned.clone();
	std::vector<std::vector<cv::Point>> full_contours_raw;
	cv::findContours(full_clone, full_contours_raw, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	const std::vector<std::vector<cv::Point>> full_contours =
		filterContoursByArea(full_contours_raw, min_contour_area);
	DUDE_ROS_DEBUGF_THROTTLE(
		2000,
		"decompose_image free_space pixels=%d extracted_contours %s min_area=%.1f",
		cv::countNonZero(free_space_mask),
		summarizeContours(full_contours).c_str(),
		min_contour_area);

	Stable_graph adjusted_state = Stable;
	adjusted_state.image_size = image_cleaned.size();
	sanitizeStableGraph(adjusted_state, image_cleaned.size(), "previous_state");
	if (current_origin_ != new_origin_) {
		cv::Point correction;
		correction.x = (current_origin_.x - new_origin_.x) / resolution;
		correction.y = (current_origin_.y - new_origin_.y) / resolution;
		applyCorrection(adjusted_state, correction);
		DUDE_ROS_DEBUGF(
			"decompose_image adjusted_contours count=%zu correction=(%d,%d)",
			adjusted_state.Region_contour.size(),
			correction.x,
			correction.y);
	} else {
		DUDE_ROS_DEBUGF(
			"decompose_image adjusted_contours count=%zu correction=(0,0)",
			adjusted_state.Region_contour.size());
	}
	
//////////////////////////////////////////////////////////
//// Decomposition
	begin_process = clock();
	(void)begin_process;

	cv::Mat stable_drawing = cv::Mat::zeros(image_cleaned.size().height, image_cleaned.size().width, CV_8UC1);
	drawContours(stable_drawing, adjusted_state.Region_contour, -1, 255, -1, 8);
	
	cv::Mat working_image;
	working_image = image_cleaned & ~stable_drawing;
//		compare(image_cleaned, stable_drawing, working_image, cv::CMP_NE);
	cv::Mat will_be_destroyed = working_image.clone();
	
	std::vector<std::vector<cv::Point> > Differential_contour;
	cv::findContours(will_be_destroyed, Differential_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		
	vector<vector<cv::Point> > big_contours_vector =
		filterContoursByArea(Differential_contour, min_contour_area);
	DUDE_ROS_DEBUGF_THROTTLE(
		2000,
		"decompose_image differential_contours extracted=%zu filtered=%zu min_area=%.1f",
		Differential_contour.size(),
		big_contours_vector.size(),
		min_contour_area);
	if (full_contours.empty()) {
		Stable = buildGraphFromContours({}, {}, image_cleaned.size());
		previous_rect = cv::Rect(0, 0, image_cleaned.cols, image_cleaned.rows);
		first_time = true;
		current_origin_ = new_origin_;
		last_branch_ = "accept_empty_free_space";
		DUDE_ROS_DEBUGF(
			"decompose_image branch=%s reason=no_valid_free_space final_regions=%zu",
			last_branch_.c_str(),
			Stable.Region_contour.size());
		return Stable;
	}
	if (big_contours_vector.empty()) {
		const double overlap_ratio =
			computeContourOverlapRatio(adjusted_state.Region_contour, free_space_mask);
		if (!adjusted_state.Region_contour.empty() && overlap_ratio >= 0.90) {
			Stable = adjusted_state;
			sanitizeStableGraph(Stable, image_cleaned.size(), "reuse_adjusted_no_differential");
			Stable.image_size = image_cleaned.size();
			previous_rect = previous_rect.area() > 0 ?
				previous_rect : cv::boundingRect(full_contours.front());
			first_time = false;
			current_origin_ = new_origin_;
			last_branch_ = "accept_reuse_adjusted_no_differential";
			DUDE_ROS_DEBUGF(
				"decompose_image branch=%s overlap=%.3f final_regions=%zu",
				last_branch_.c_str(),
				overlap_ratio,
				Stable.Region_contour.size());
			return Stable;
		}

		last_rejection_reason_ =
			"differential_contours_empty_with_valid_free_space";
		last_branch_ = "fallback_full_decomposition";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s adjusted_regions=%zu overlap=%.3f",
			last_branch_.c_str(),
			last_rejection_reason_.c_str(),
			adjusted_state.Region_contour.size(),
			overlap_ratio);

		WrapperRunResult wrapper =
			runWrapperInChild(image_cleaned, pixel_Tau, "fallback_full_no_differential");
		resize_rect = wrapper.resize_rect;
		Stable_graph candidate = buildGraphFromContours(
			wrapper.decomposed_contours, wrapper.contours_centroid, image_cleaned.size());
		DUDE_ROS_DEBUGF(
			"decompose_image fallback wrappers=1 input_contours=%zu polygons=%zu rejected_polygons=%zu cuts_after=%zu raw_decomposed_regions=%zu final_regions=%zu",
			wrapper.input_contours,
			wrapper.valid_polygons,
			wrapper.rejected_polygons,
			wrapper.cuts_after_resolution,
			wrapper.decomposed_contours.size(),
			candidate.Region_contour.size());
		if (wrapper.ok && !candidate.Region_contour.empty()) {
			Stable = candidate;
			sanitizeStableGraph(Stable, image_cleaned.size(), "fallback_full_no_differential");
			previous_rect = resize_rect.area() > 0 ?
				resize_rect : cv::boundingRect(full_contours.front());
			first_time = false;
			current_origin_ = new_origin_;
			return Stable;
		}

		if (!adjusted_state.Region_contour.empty()) {
			last_rejection_reason_ += ";fallback_empty_preserving_previous";
			Stable = adjusted_state;
			sanitizeStableGraph(Stable, image_cleaned.size(), "fallback_empty_preserve_previous");
			Stable.image_size = image_cleaned.size();
			previous_rect = previous_rect.area() > 0 ?
				previous_rect : cv::boundingRect(full_contours.front());
			first_time = false;
			current_origin_ = new_origin_;
			last_branch_ = "reject_empty_fallback_preserve_previous";
			DUDE_ROS_WARNF(
				"decompose_image branch=%s reason=%s final_regions=%zu",
				last_branch_.c_str(),
				last_rejection_reason_.c_str(),
				Stable.Region_contour.size());
			return Stable;
		}

		Stable = buildGraphFromContours(full_contours, {}, image_cleaned.size());
		sanitizeStableGraph(Stable, image_cleaned.size(), "free_contours_fallback");
		previous_rect = cv::boundingRect(full_contours.front());
		first_time = false;
		current_origin_ = new_origin_;
		last_branch_ = "accept_free_contours_fallback";
		last_rejection_reason_ += ";wrapper_empty_using_free_contours";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s final_regions=%zu",
			last_branch_.c_str(),
			last_rejection_reason_.c_str(),
			Stable.Region_contour.size());
		return Stable;
	}
	if(first_time) resize_rect = cv::boundingRect(big_contours_vector[0]);
	else resize_rect= previous_rect;

	vector<vector<cv::Point> > connected_contours, unconnected_contours;
	std::vector<cv::Point> unconnected_centroids, connected_centroids;
	(void)connected_centroids;
	vector< vector <int > > conection_prev_new;
	for(int i=0;i < static_cast<int>(adjusted_state.Region_contour.size());i++){
		bool is_stable_connected = false;
		for(int j=0;j < static_cast<int>(big_contours_vector.size());j++){
			int connected = 0;
			cv::Point centroid;
			are_contours_connected(adjusted_state.Region_contour[i], big_contours_vector[j] , centroid, connected);
			if (connected>0){
				vector <int> pair;
				pair.push_back(i);
				pair.push_back(j);
				conection_prev_new.push_back(pair);
				is_stable_connected = true;
			}
		}
		if(is_stable_connected == false){
			unconnected_contours.push_back(adjusted_state.Region_contour[i]);
			unconnected_centroids.push_back(
				i < static_cast<int>(adjusted_state.Region_centroid.size()) ?
				adjusted_state.Region_centroid[i] :
				contourCentroidOrFirst(adjusted_state.Region_contour[i]));
		}
		else{
			connected_contours.push_back(adjusted_state.Region_contour[i]);
			connected_centroids.push_back(
				i < static_cast<int>(adjusted_state.Region_centroid.size()) ?
				adjusted_state.Region_centroid[i] :
				contourCentroidOrFirst(adjusted_state.Region_contour[i]));
		}
	}
	DUDE_ROS_DEBUGF(
		"decompose_image matched_contours matches=%zu connected=%zu unconnected=%zu",
		conection_prev_new.size(),
		connected_contours.size(),
		unconnected_contours.size());

	cv::Mat expanded_drawing = cv::Mat::zeros(image_cleaned.size().height, image_cleaned.size().width, CV_8UC1);
	drawContours(expanded_drawing, connected_contours,  -1, 2, -1, 8);

	for(int i=0; i < static_cast<int>(conection_prev_new.size());i++){
		drawContours(expanded_drawing, big_contours_vector, conection_prev_new[i][1] , 2, -1, 8);
	}


	will_be_destroyed = expanded_drawing.clone();			
	std::vector<std::vector<cv::Point> > Expanded_contour;
	cv::findContours(will_be_destroyed, Expanded_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

	if(first_time){
		Expanded_contour.clear();
		Expanded_contour = big_contours_vector;
		first_time=false;
	}
	DUDE_ROS_DEBUGF(
		"decompose_image expanded_contours %s",
		summarizeContours(Expanded_contour).c_str());
	if (Expanded_contour.empty()) {
		last_branch_ = "fallback_full_decomposition";
		last_rejection_reason_ = "expanded_contours_empty";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s",
			last_branch_.c_str(),
			last_rejection_reason_.c_str());
		if (!adjusted_state.Region_contour.empty()) {
			last_branch_ = "reject_expanded_empty_preserve_previous";
			last_rejection_reason_ += ";skipped_unsafe_full_fallback";
			Stable = adjusted_state;
			sanitizeStableGraph(Stable, image_cleaned.size(), "expanded_empty_preserve_previous_before_full_fallback");
			previous_rect = previous_rect.area() > 0 ?
				previous_rect : cv::boundingRect(full_contours.front());
			first_time = false;
			current_origin_ = new_origin_;
			DUDE_ROS_WARNF(
				"decompose_image branch=%s reason=%s final_regions=%zu kept_previous=true",
				last_branch_.c_str(),
				last_rejection_reason_.c_str(),
				Stable.Region_contour.size());
			return Stable;
		}
		Stable = buildGraphFromContours(full_contours, {}, image_cleaned.size());
		previous_rect = cv::boundingRect(full_contours.front());
		first_time = false;
		current_origin_ = new_origin_;
		last_branch_ = "accept_free_contours_fallback";
		last_rejection_reason_ += ";no_previous_state_skipped_unsafe_full_fallback";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s final_regions=%zu kept_previous=false",
			last_branch_.c_str(),
			last_rejection_reason_.c_str(),
			Stable.Region_contour.size());
		return Stable;
	}

	begin_process = clock();
	(void)begin_process;

	vector<WrapperRunResult> wrapper_vector;
	wrapper_vector.reserve(Expanded_contour.size());
	size_t raw_decomposed_regions = 0;
	for(int i = 0; i < static_cast<int>(Expanded_contour.size());i++){
		cv::Mat Temporal_Image = cv::Mat::zeros(image_cleaned.size().height, image_cleaned.size().width, CV_8UC1);								
		cv::Mat temporal_image_cut = cv::Mat::zeros(image_cleaned.size().height, image_cleaned.size().width, CV_8UC1);								
		drawContours(Temporal_Image, Expanded_contour, i, 255, -1, 8);
		image_cleaned.copyTo(temporal_image_cut,Temporal_Image);
		
		WrapperRunResult wrapper =
			runWrapperInChild(temporal_image_cut, pixel_Tau, "incremental_expanded_contour");
		if (wrapper.ok) {
			resize_rect |= wrapper.resize_rect;
			raw_decomposed_regions += wrapper.decomposed_contours.size();
		}
		wrapper_vector.push_back(std::move(wrapper));
	}	
	size_t wrapper_input_contours = 0;
	size_t wrapper_valid_polygons = 0;
	size_t wrapper_rejected_polygons = 0;
	size_t wrapper_cuts_after_resolution = 0;
	for (const auto &wrapper : wrapper_vector) {
		wrapper_input_contours += wrapper.input_contours;
		wrapper_valid_polygons += wrapper.valid_polygons;
		wrapper_rejected_polygons += wrapper.rejected_polygons;
		wrapper_cuts_after_resolution += wrapper.cuts_after_resolution;
	}
	DUDE_ROS_DEBUGF(
		"decompose_image wrappers=%zu expanded_contours=%zu input_contours=%zu polygons=%zu rejected_polygons=%zu cuts_after=%zu raw_decomposed_regions=%zu",
		wrapper_vector.size(),
		Expanded_contour.size(),
		wrapper_input_contours,
		wrapper_valid_polygons,
		wrapper_rejected_polygons,
		wrapper_cuts_after_resolution,
		raw_decomposed_regions);

	vector<vector<cv::Point> > joint_contours = unconnected_contours;
	vector<cv::Point> joint_centroids = unconnected_centroids;
	
	for(int i = 0; i < static_cast<int>(wrapper_vector.size());i++){
		if (!wrapper_vector[i].ok) {
			continue;
		}
		for(int j = 0; j < static_cast<int>(wrapper_vector[i].decomposed_contours.size());j++){
			if (!isValidStableContour(wrapper_vector[i].decomposed_contours[j])) {
				continue;
			}
			joint_contours.push_back(wrapper_vector[i].decomposed_contours[j]);
			joint_centroids.push_back(
				j < static_cast<int>(wrapper_vector[i].contours_centroid.size()) ?
				wrapper_vector[i].contours_centroid[j] :
				contourCentroidOrFirst(wrapper_vector[i].decomposed_contours[j]));
		}
	}	
	

///////////////////
//// Build stable graph
	begin_process = clock();
	(void)begin_process;

	if (joint_contours.empty()) {
		last_branch_ = "fallback_full_decomposition";
		last_rejection_reason_ = "joint_contours_empty";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s",
			last_branch_.c_str(),
			last_rejection_reason_.c_str());
		WrapperRunResult wrapper =
			runWrapperInChild(image_cleaned, pixel_Tau, "fallback_full_joint_empty");
		resize_rect = wrapper.resize_rect;
		Stable_graph candidate = buildGraphFromContours(
			wrapper.decomposed_contours, wrapper.contours_centroid, image_cleaned.size());
		DUDE_ROS_DEBUGF(
			"decompose_image fallback wrappers=1 reason=%s input_contours=%zu polygons=%zu rejected_polygons=%zu cuts_after=%zu raw_decomposed_regions=%zu final_regions=%zu",
			last_rejection_reason_.c_str(),
			wrapper.input_contours,
			wrapper.valid_polygons,
			wrapper.rejected_polygons,
			wrapper.cuts_after_resolution,
			wrapper.decomposed_contours.size(),
			candidate.Region_contour.size());
		if (wrapper.ok && !candidate.Region_contour.empty()) {
			Stable = candidate;
			sanitizeStableGraph(Stable, image_cleaned.size(), "fallback_full_joint_empty");
			previous_rect = resize_rect;
			first_time = false;
			current_origin_ = new_origin_;
			DUDE_ROS_DEBUGF(
				"decompose_image branch=%s final_regions=%zu",
				last_branch_.c_str(),
				Stable.Region_contour.size());
			return Stable;
		}

		Stable = adjusted_state;
		sanitizeStableGraph(Stable, image_cleaned.size(), "joint_empty_preserve_previous");
		Stable.image_size = image_cleaned.size();
		previous_rect = previous_rect.area() > 0 ? previous_rect : resize_rect;
		first_time = Stable.Region_contour.empty();
		current_origin_ = new_origin_;
		last_branch_ = "reject_empty_candidate_preserve_previous";
		last_rejection_reason_ += ";fallback_empty";
		DUDE_ROS_WARNF(
			"decompose_image branch=%s reason=%s final_regions=%zu",
			last_branch_.c_str(),
			last_rejection_reason_.c_str(),
			Stable.Region_contour.size());
		return Stable;
	}

	Stable.Region_contour  = joint_contours;
	Stable.Region_centroid = joint_centroids;
	Stable.image_size = image_cleaned.size();
	sanitizeStableGraph(Stable, image_cleaned.size(), "accept_incremental");
	previous_rect = resize_rect;
	current_origin_ = new_origin_;
	first_time = false;
	last_branch_ = "accept_incremental";
	DUDE_ROS_DEBUGF(
		"decompose_image branch=%s final_regions=%zu",
		last_branch_.c_str(),
		Stable.Region_contour.size());
	
	return Stable;
}


/////////////////////////////
//// UTILITIES

void Incremental_Decomposer::are_contours_connected(const vector<cv::Point>& first_contour, const vector<cv::Point>& second_contour, cv::Point &centroid, int &number_of_ones ){
	
	vector< cv::Point > closer_point;
	cv::Point acum(0,0);
	int threshold=2;
	
	for(int i=0; i<static_cast<int>(first_contour.size());i++){
		for(int j=0; j< static_cast<int>(second_contour.size());j++){
			float distance;
			distance = cv::norm(first_contour[i] -  second_contour[j] );
			if(distance < threshold){
				cv::Point point_to_add;
				point_to_add.x = (first_contour[i].x + second_contour[j].x)/2;
				point_to_add.y = (first_contour[i].y + second_contour[j].y)/2;
				
				closer_point.push_back(point_to_add);
				acum += point_to_add;						
			 }					
		}
	}

	number_of_ones = closer_point.size();
	if(number_of_ones>0){
		centroid.x = acum.x/number_of_ones;
		centroid.y = acum.y/number_of_ones;
	}
}
	
void Incremental_Decomposer::adjust_stable_contours(){
	cv::Point correction;
	
// considering constant resolution
	correction.x = (current_origin_.x - new_origin_.x) / resolution;
	correction.y = (current_origin_.y - new_origin_.y) / resolution;
	
	sanitizeStableGraph(Stable, Stable.image_size, "adjust_stable_contours");
	for(int i=0; i < static_cast<int>(Stable.Region_contour.size());i++){
		Stable.Region_centroid[i] += correction;
		for(int j=0; j < static_cast<int>(Stable.Region_contour[i].size());j++){
			Stable.Region_contour[i][j] += correction;
//					Stable.Region_contour[i][j] = cartesian_to_pixel(pixel_to_cartesian(Stable.Region_contour[i][j]));
		}
	}
		
	current_origin_ = new_origin_;
}

cv::Point Incremental_Decomposer::pixel_to_cartesian(cv::Point point_in){
	cv::Point point_out;
	point_out.x = point_in.x * resolution + current_origin_.x;
	point_out.y = point_in.y * resolution + current_origin_.y;
	
	return point_out;
}

cv::Point Incremental_Decomposer::cartesian_to_pixel(cv::Point point_in){
	cv::Point point_out;
	point_out.x = (point_in.x - new_origin_.x) / resolution ;
	point_out.y = (point_in.y - new_origin_.y) / resolution ;			
	
	return point_out;
}
