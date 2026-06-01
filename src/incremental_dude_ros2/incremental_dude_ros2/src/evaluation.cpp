// ROS2
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

// openCV
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

// DuDe
#include "inc_decomp.hpp"

// cpp
#include <dirent.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct results{
	double time;
	double precision;
	double recall;
};

class ROS_handler : public rclcpp::Node
{
	image_transport::Publisher image_pub_, image_pub2_, image_pub3_;
	std::shared_ptr<image_transport::ImageTransport> image_transport_;
	cv_bridge::CvImagePtr cv_ptr_, cv_ptr2_, cv_ptr3_;

	rclcpp::TimerBase::SharedPtr timer_;
	rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;

	float Decomp_threshold_;
	bool segmentation_ready;
	std::vector<double> clean_time_vector, decomp_time_vector, paint_time_vector, complete_time_vector;

	std::string  base_path;
	std::string gt_ending;
	std::string FuT_ending;
	std::string No_FuT_ending;

	std::vector< std::vector<float> > Precisions;
	std::vector< std::vector<float> > Recalls;
	std::vector<double> Times;

	int current_file;
	std::set<std::string>  file_list;
	std::set<std::string>::iterator file_it;

	double pixel_precision, pixel_recall;

public:
	explicit ROS_handler(float threshold)
	: Node("evaluation"), Decomp_threshold_(threshold)
	{
		timer_ = this->create_wall_timer(
			std::chrono::milliseconds(500),
			std::bind(&ROS_handler::metronomeCallback, this));
		twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
			"cmd_vel", rclcpp::QoS(1),
			std::bind(&ROS_handler::twistCallback, this, std::placeholders::_1));
		segmentation_ready = false;

		base_path = "src/incremental_dude_ros2/incremental_dude_ros2/maps/Room_Segmentation/all_maps";
		gt_ending = "_gt_segmentation.png";
		FuT_ending = "_furnitures.png";
		No_FuT_ending = ".png";

		current_file = 0;
		file_list = listFile();
		file_it = file_list.begin();

		if(file_it != file_list.end()){
			RCLCPP_INFO(this->get_logger(), "Ready to process file '%s'", file_it->c_str());
		}
	}

	void init_publishers()
	{
		image_transport_ =
			std::make_shared<image_transport::ImageTransport>(shared_from_this());

		image_pub_  = image_transport_->advertise("/ground_truth_segmentation", 1);
		image_pub2_ = image_transport_->advertise("/DuDe_segmentation", 1);
		image_pub3_ = image_transport_->advertise("/Inc_DuDe_segmentation", 1);

		cv_ptr_.reset(new cv_bridge::CvImage);
		cv_ptr_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;

		cv_ptr2_.reset(new cv_bridge::CvImage);
		cv_ptr2_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;

		cv_ptr3_.reset(new cv_bridge::CvImage);
		cv_ptr3_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
	}

private:
	typedef std::map<std::vector<int>, std::vector<cv::Point> > match2points;
	typedef std::map<int, std::vector<cv::Point> > tag2points;
	typedef std::map<int, tag2points> tag2tagMapper;

	double getTime() const
	{
		return std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	void metronomeCallback()
	{
		if(segmentation_ready) publish_Image();
	}

	void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
	{
		float direction = msg->linear.x;

		if(direction > 0){
			file_it++;
			if(file_it == file_list.end()) file_it = file_list.begin();

			if(file_it != file_list.end()){
				RCLCPP_INFO(this->get_logger(), "Selected file '%s'", file_it->c_str());
			}
		}
		else{
			if(file_it == file_list.end()){
				return;
			}

			RCLCPP_INFO(this->get_logger(), "Processing file '%s'", file_it->c_str());
			Precisions.clear();
			Recalls.clear();

			process_files_twice(*file_it);

			RCLCPP_INFO(this->get_logger(), "Finished file '%s'", file_it->c_str());

			publish_Image();
		}
	}

	void publish_Image()
	{
		if(cv_ptr_ && !cv_ptr_->image.empty()){
			image_pub_.publish(cv_ptr_->toImageMsg());
		}
		if(cv_ptr2_ && !cv_ptr2_->image.empty()){
			image_pub2_.publish(cv_ptr2_->toImageMsg());
		}
		if(cv_ptr3_ && !cv_ptr3_->image.empty()){
			image_pub3_.publish(cv_ptr3_->toImageMsg());
		}
	}

	void process_files(std::string name){
		cv::Mat image_GT, image_Furniture, image_No_Furniture;
		cv::Mat DuDe_Furniture, DuDe_No_Furniture;
		cv::Mat proxy, zero_image;

		std::string full_path_GT           = base_path + "/" + name + gt_ending;
		std::string full_path_Furniture    = base_path + "/" + name + FuT_ending;
		std::string full_path_No_Furniture = base_path + "/" + name + No_FuT_ending;
		std::string saving_path            = base_path + "/Tagged_Images/" + name;

		double begin_process, end_process, decompose_time;
		std::vector<cv::Vec3b> colormap;
		results No_Furn_Results_pixel, No_Furn_Results_Regions;
		results Furn_Results_pixel, Furn_Results_Regions;

		image_GT           = cv::imread(full_path_GT,0);
		image_Furniture    = cv::imread(full_path_Furniture,0);
		image_No_Furniture = cv::imread(full_path_No_Furniture,0);
		zero_image = cv::Mat::zeros(image_GT.size(),CV_8U);

		cv::Mat GT_segmentation = segment_Ground_Truth(image_GT);
		colormap = save_image_original_color(saving_path + "_TAG" + gt_ending, GT_segmentation);

		begin_process = getTime();
		DuDe_No_Furniture = simple_segment(image_No_Furniture);
		end_process = getTime();	decompose_time = end_process - begin_process;
		compare_images(GT_segmentation, DuDe_No_Furniture);
		std::map<int,int> DuDe_NoF_map = compare_images2(GT_segmentation, DuDe_No_Furniture);

		DuDe_No_Furniture.copyTo(proxy ,image_No_Furniture>250);
		DuDe_No_Furniture = proxy.clone();

		No_Furn_Results_pixel.time = No_Furn_Results_Regions.time = decompose_time;
		extract_results(No_Furn_Results_pixel, No_Furn_Results_Regions);

		begin_process = getTime();
		DuDe_Furniture = simple_segment(image_Furniture);
		end_process = getTime();	decompose_time = end_process - begin_process;
		compare_images(GT_segmentation, DuDe_Furniture);
		std::map<int,int> DuDe_Furn_map = compare_images2(GT_segmentation, DuDe_Furniture);

		DuDe_Furniture.copyTo(proxy ,image_Furniture>250);
		DuDe_Furniture = proxy.clone();

		Furn_Results_pixel.time = Furn_Results_Regions.time = decompose_time;
		extract_results(Furn_Results_pixel, Furn_Results_Regions);

		double min, max;
		cv::minMaxLoc(GT_segmentation,&min,&max);

		float rows = GT_segmentation.rows;
		float cols = GT_segmentation.cols;
		float proper_size = rows*cols/1000;
		proper_size = proper_size/1000;

		RCLCPP_INFO(
			this->get_logger(),
			"%s no_furniture precision=%.1f recall=%.1f time_ms=%.0f labels=%.0f size_km2=%.2f",
			name.c_str(),
			No_Furn_Results_Regions.precision,
			No_Furn_Results_Regions.recall,
			No_Furn_Results_Regions.time,
			max,
			proper_size);

		RCLCPP_INFO(
			this->get_logger(),
			"%s furniture precision=%.1f recall=%.1f time_ms=%.0f labels=%.0f size_km2=%.2f",
			name.c_str(),
			Furn_Results_Regions.precision,
			Furn_Results_Regions.recall,
			Furn_Results_Regions.time,
			max,
			proper_size);

		cv::Mat to_publish = image_GT.clone();
		cv_ptr_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish.convertTo(to_publish, CV_32F);
		to_publish.copyTo(cv_ptr_->image);

		cv::Mat to_publish2 = image_Furniture.clone();
		cv_ptr2_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish2.convertTo(to_publish2, CV_32F);
		to_publish2.copyTo(cv_ptr2_->image);

		cv::Mat to_publish3 = image_No_Furniture.clone();
		cv_ptr3_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish2.convertTo(to_publish3, CV_32F);
		to_publish3.copyTo(cv_ptr3_->image);

		segmentation_ready = true;
	}

	void process_files_incrementally(std::string name){
		cv::Mat image_GT, image_Furniture, image_No_Furniture;
		cv::Mat Inc_Furniture, Inc_No_Furniture;
		cv::Mat proxy, zero_image;

		std::string full_path_GT           = base_path + "/" + name + gt_ending;
		std::string full_path_Furniture    = base_path + "/" + name + FuT_ending;
		std::string full_path_No_Furniture = base_path + "/" + name + No_FuT_ending;
		std::string saving_path       = base_path + "/Tagged_Images/" + name;

		double begin_process, end_process, decompose_time;
		std::vector<cv::Vec3b> colormap;
		results No_Furn_Results_pixel, No_Furn_Results_Regions;
		results Furn_Results_pixel, Furn_Results_Regions;

		image_GT           = cv::imread(full_path_GT,0);
		image_Furniture    = cv::imread(full_path_Furniture,0);
		image_No_Furniture = cv::imread(full_path_No_Furniture,0);
		zero_image =cv::Mat::zeros(image_GT.size(),CV_8U);

		cv::Mat GT_segmentation = segment_Ground_Truth(image_GT);
		colormap = save_image_original_color(saving_path + "_TAG_inc" + gt_ending, GT_segmentation);

		Inc_No_Furniture = incremental_segment(image_No_Furniture, decompose_time);
		compare_images(GT_segmentation, Inc_No_Furniture);
		std::map<int,int> DuDe_NoF_map = compare_images2(GT_segmentation, Inc_No_Furniture);

		Inc_No_Furniture.copyTo(proxy ,image_No_Furniture>250);
		Inc_No_Furniture = proxy.clone();

		No_Furn_Results_pixel.time = No_Furn_Results_Regions.time = decompose_time;
		extract_results(No_Furn_Results_pixel, No_Furn_Results_Regions);

		Inc_Furniture = incremental_segment(image_Furniture, decompose_time);
		compare_images(GT_segmentation, Inc_Furniture);
		std::map<int,int> DuDe_Furn_map = compare_images2(GT_segmentation, Inc_Furniture);

		Inc_Furniture.copyTo(proxy ,image_Furniture>250);
		Inc_Furniture = proxy.clone();

		Furn_Results_pixel.time = Furn_Results_Regions.time = decompose_time;
		extract_results(Furn_Results_pixel, Furn_Results_Regions);

		double min, max;
		cv::minMaxLoc(GT_segmentation,&min,&max);

		float rows = GT_segmentation.rows;
		float cols = GT_segmentation.cols;
		float proper_size = rows*cols/1000;
		proper_size = proper_size/1000;

		RCLCPP_INFO(
			this->get_logger(),
			"%s no_furniture precision=%.1f recall=%.1f time_ms=%.0f labels=%.0f size_km2=%.2f",
			name.c_str(),
			No_Furn_Results_Regions.precision,
			No_Furn_Results_Regions.recall,
			No_Furn_Results_Regions.time,
			max,
			proper_size);

		RCLCPP_INFO(
			this->get_logger(),
			"%s furniture precision=%.1f recall=%.1f time_ms=%.0f labels=%.0f size_km2=%.2f",
			name.c_str(),
			Furn_Results_Regions.precision,
			Furn_Results_Regions.recall,
			Furn_Results_Regions.time,
			max,
			proper_size);

		cv::Mat to_publish = image_GT.clone();
		cv_ptr_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish.convertTo(to_publish, CV_32F);
		to_publish.copyTo(cv_ptr_->image);

		cv::Mat to_publish2 = image_Furniture.clone();
		cv_ptr2_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish2.convertTo(to_publish2, CV_32F);
		to_publish2.copyTo(cv_ptr2_->image);

		cv::Mat to_publish3 = image_No_Furniture.clone();
		cv_ptr3_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish2.convertTo(to_publish3, CV_32F);
		to_publish3.copyTo(cv_ptr3_->image);

		segmentation_ready = true;
	}

	void process_files_twice(std::string name){
		cv::Mat image_GT, image_Furniture, image_No_Furniture;
		cv::Mat Inc_Furniture, DuDe_Furniture;
		cv::Mat proxy, zero_image;

		std::string full_path_GT           = base_path + "/" + name + gt_ending;
		std::string full_path_Furniture    = base_path + "/" + name + FuT_ending;
		std::string full_path_No_Furniture = base_path + "/" + name + No_FuT_ending;
		std::string saving_path       = base_path + "/Tagged_Images/" + name;

		double begin_process, end_process, decompose_time;
		std::vector<cv::Vec3b> colormap;
		results No_Furn_Results_pixel, No_Furn_Results_Regions;
		results Furn_Results_pixel, Furn_Results_Regions;

		image_GT           = cv::imread(full_path_GT,0);
		image_Furniture    = cv::imread(full_path_Furniture,0);
		zero_image =cv::Mat::zeros(image_GT.size(),CV_8U);

		cv::Mat GT_segmentation = segment_Ground_Truth(image_GT);
		colormap = save_image_original_color(saving_path + "_TAG" + gt_ending, GT_segmentation);

		DuDe_Furniture = simple_segment(image_Furniture);
		compare_images(GT_segmentation, DuDe_Furniture);
		std::map<int,int> DuDe_Furn_map = compare_images(GT_segmentation, DuDe_Furniture);

		DuDe_Furniture.copyTo(proxy , image_Furniture>250);
		DuDe_Furniture = proxy.clone();
		save_decomposed_image_color(saving_path + "_DuDe" + FuT_ending, DuDe_Furniture, colormap, DuDe_Furn_map);  proxy = zero_image;

		No_Furn_Results_pixel.time = No_Furn_Results_Regions.time = decompose_time;
		extract_results(No_Furn_Results_pixel, No_Furn_Results_Regions);

		Inc_Furniture = incremental_segment(image_Furniture, decompose_time);
		compare_images(GT_segmentation, Inc_Furniture);
		std::map<int,int> Inc_Furn_map = compare_images(GT_segmentation, Inc_Furniture);

		Inc_Furniture.copyTo(proxy ,image_Furniture>250);
		Inc_Furniture = proxy.clone();
		save_decomposed_image_color(saving_path + "_Inc" + FuT_ending, Inc_Furniture, colormap, Inc_Furn_map);  proxy = zero_image;

		Furn_Results_pixel.time = Furn_Results_Regions.time = decompose_time;
		extract_results(Furn_Results_pixel, Furn_Results_Regions);

		double min, max;
		cv::minMaxLoc(GT_segmentation,&min,&max);

		float rows = GT_segmentation.rows;
		float cols = GT_segmentation.cols;
		float proper_size = rows*cols/1000;
		proper_size = proper_size/1000;

		RCLCPP_INFO(
			this->get_logger(),
			"%s no_furniture precision=%.1f recall=%.1f time_ms=%.0f labels=%.0f size_km2=%.2f",
			name.c_str(),
			No_Furn_Results_Regions.precision,
			No_Furn_Results_Regions.recall,
			No_Furn_Results_Regions.time,
			max,
			proper_size);

		RCLCPP_INFO(
			this->get_logger(),
			"%s furniture precision=%.1f recall=%.1f time_ms=%.0f labels=%.0f size_km2=%.2f",
			name.c_str(),
			Furn_Results_Regions.precision,
			Furn_Results_Regions.recall,
			Furn_Results_Regions.time,
			max,
			proper_size);

		cv::Mat to_publish = image_GT.clone();
		cv_ptr_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish.convertTo(to_publish, CV_32F);
		to_publish.copyTo(cv_ptr_->image);

		cv::Mat to_publish2 = image_Furniture.clone();
		cv_ptr2_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish2.convertTo(to_publish2, CV_32F);
		to_publish2.copyTo(cv_ptr2_->image);

		cv::Mat to_publish3 = image_No_Furniture.clone();
		cv_ptr3_->encoding = sensor_msgs::image_encodings::TYPE_32FC1;			to_publish2.convertTo(to_publish3, CV_32F);
		to_publish3.copyTo(cv_ptr3_->image);

		segmentation_ready = true;
	}

	std::map<int,int> compare_images(cv::Mat GT_segmentation_in, cv::Mat DuDe_segmentation_in){
		std::map<int,int> segmented2GT_tags;

		cv::Mat GT_segmentation   = cv::Mat::zeros(GT_segmentation_in.size(),CV_8UC1);
		cv::Mat DuDe_segmentation = cv::Mat::zeros(GT_segmentation_in.size(),CV_8UC1);

		GT_segmentation_in  .convertTo(GT_segmentation, CV_8UC1);
		DuDe_segmentation_in.convertTo(DuDe_segmentation, CV_8UC1);
		tag2tagMapper gt_tag2mapper,DuDe_tag2mapper;

		for(int x=0; x < GT_segmentation.size().width; x++){
			for(int y=0; y < GT_segmentation.size().height; y++){
				cv::Point current_pixel(x,y);

				int tag_GT   = GT_segmentation.at<uchar>(current_pixel);
				int tag_DuDe  = DuDe_segmentation.at<uchar>(current_pixel);

				if(tag_DuDe>0 && tag_GT>0 ){
					gt_tag2mapper  [tag_GT][tag_DuDe].push_back(current_pixel);
					DuDe_tag2mapper[tag_DuDe][tag_GT].push_back(current_pixel);
				}
			}
		}

		std::vector<float> precisions_inside, recalls_inside;
		double cum_precision=0, cum_total=0, cum_recall=0;

		for(tag2tagMapper::iterator it = gt_tag2mapper.begin(); it!= gt_tag2mapper.end(); it++ ){
			tag2points inside = it->second;
			int max_intersection=0, total_points=0;
			int gt_tag_max = -1;
			for(tag2points::iterator it2 = inside.begin(); it2!= inside.end(); it2++ ){
				total_points += it2->second.size();
				if (static_cast<int>(it2->second.size()) > max_intersection){
					max_intersection = it2->second.size();
					gt_tag_max = it2->first;
				}
			}
			segmented2GT_tags[gt_tag_max] = it->first;
			precisions_inside.push_back(100*max_intersection/total_points);
			cum_precision += max_intersection;
			cum_total += total_points;
		}
		pixel_precision = cum_precision/cum_total;

		cum_total=0;
		for(tag2tagMapper::iterator it = DuDe_tag2mapper.begin(); it!= DuDe_tag2mapper.end(); it++ ){
			tag2points inside = it->second;
			int max_intersection=0, total_points=0;
			for(tag2points::iterator it2 = inside.begin(); it2!= inside.end(); it2++ ){
				total_points += it2->second.size();
				if (static_cast<int>(it2->second.size()) > max_intersection) max_intersection = it2->second.size();
			}
			recalls_inside.push_back(100*max_intersection/total_points);
			cum_recall += max_intersection;
			cum_total += total_points;
		}
		pixel_recall=cum_recall/cum_total;

		Precisions.push_back(precisions_inside);
		Recalls.push_back(recalls_inside);
		return segmented2GT_tags;
	}

	std::map<int,int> compare_images2(cv::Mat GT_segmentation_in, cv::Mat DuDe_segmentation_in){
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

		std::map<std::vector<int>, float> link2relation;
		std::map<int,int> DuDe_Union_Match;
		int current_DuDe_Tag=0;
		int current_GT_max = -1;
		int current_DuDe_evaluated = -1;
		float current_GT_max_relation = -1;

		for(match2points::iterator it = links2points.begin(); it!= links2points.end(); it++ ){
			std::vector<cv::Point> points_in_match = it->second;

			float A = GT_points  [ it->first[1] ].size();
			float B = DuDe_points[ it->first[0] ].size();
			float AandB = points_in_match.size();
			float relation = AandB/( A + B - AandB );

			link2relation[it->first] = relation;

			if(current_DuDe_evaluated != it->first[0] ){
				DuDe_Union_Match[current_DuDe_evaluated] = current_GT_max;
				current_DuDe_evaluated = it->first[0];
				current_GT_max = it->first[1];
				current_GT_max_relation = relation;
			}
			else if(relation > current_GT_max_relation){
				current_GT_max = it->first[1];
				current_GT_max_relation = relation;
			}
		}
		DuDe_Union_Match[current_DuDe_evaluated] = current_GT_max;
		return DuDe_Union_Match;
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

	cv::Mat segment_Ground_Truth(cv::Mat GroundTruth_BW){
		cv::Mat src = GroundTruth_BW.clone();
		cv::Mat drawing = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

		src = src > 250;

		cv::erode(src, src, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

		int idx = 0;
		int color=1;
		for(; idx >= 0; idx = hierarchy[idx][0]){
			cv::drawContours(drawing, contours, idx, color , cv::FILLED, 20, hierarchy);
			color++;
		}
		cv::dilate(drawing, drawing, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
		return drawing;
	}

	cv::Mat incremental_segment(cv::Mat image_in, double & time){
		Incremental_Decomposer inc_decomp;
		Stable_graph Stable;
		cv::Point2f origin(0,0);
		float resolution = 0.05;

		cv::Mat pre_decompose = image_in.clone();
		cv::Mat pre_decompose_BW = pre_decompose > 250;

		cv::Mat current_circle = cv::Mat::zeros(image_in.size(),CV_8UC1);
		cv::Mat scanned_image = cv::Mat::zeros(image_in.size(),CV_8UC1);
		cv::Mat current_scan;

		int counter=0;
		bool stop_criteria=false;
		float cum_time=0;
		int valid_images=0;

		while(!stop_criteria){
			div_t divresult = div(50*counter,pre_decompose_BW.size().width);
			int x = divresult.rem;
			int y = 50*divresult.quot;

			cv::Point current_position = cv::Point(x,y);
			cv::circle(current_circle, current_position, 100, 1, -1);

			pre_decompose_BW.copyTo(current_scan, current_circle);

			if (cv::countNonZero(current_scan)  > 160) {
				double begin_process, end_process, decompose_time;
				begin_process = getTime();
				Stable = inc_decomp.decompose_image(current_scan, Decomp_threshold_/resolution, origin , resolution);
				end_process = getTime();	decompose_time = end_process - begin_process;
				valid_images++;
				cum_time += decompose_time;
			}

			if((x > pre_decompose_BW.size().width) || (y > pre_decompose_BW.size().height)) {
				stop_criteria = true;
			}
			scanned_image |= current_scan;
			counter++;
		}

		cv::Mat Drawing = cv::Mat::zeros(image_in.size(), CV_8UC1);
		for(int i = 0; i < static_cast<int>(Stable.Region_contour.size());i++){
			drawContours(Drawing, Stable.Region_contour, i, i+1, -1, 8);
		}
		time = cum_time/valid_images;
		RCLCPP_DEBUG(
			this->get_logger(),
			"Incremental segmentation produced %zu regions",
			Stable.Region_contour.size());

		return Drawing;
	}

	std::set<std::string> listFile(){
		DIR *pDIR;
		struct dirent *entry;
		std::set<std::string> files_to_read;

		if((pDIR = opendir(base_path.c_str()))){
			while((entry = readdir(pDIR))){
				if(std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0){
					std::string const fullString = entry->d_name;

					if (fullString.length() >= gt_ending.length()) {
						if (0 == fullString.compare(fullString.length() - gt_ending.length(), gt_ending.length(), gt_ending)){
							std::string filename= fullString.substr(0,fullString.length() - gt_ending.length());
							files_to_read.insert(filename);
						}
					}
				}
			}
			closedir(pDIR);
		}
		RCLCPP_INFO(this->get_logger(), "Found %zu evaluation files", files_to_read.size());
		return files_to_read;
	}

	void save_decomposed_image_color(std::string path, cv::Mat image_in, std::vector<cv::Vec3b> colormap, std::map<int,int> original_map){
		double min, max;
		std::vector<cv::Vec3b> color_vector;
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
				cv::Vec3b color(std::rand() % 255,std::rand() % 255,std::rand() % 255);
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

	std::vector<cv::Vec3b> save_image_original_color(std::string path, cv::Mat image_in){
		double min, max;
		std::vector<cv::Vec3b> color_vector;
		cv::Vec3b black(208, 208, 208);
		color_vector.push_back(black);

		cv::minMaxLoc(image_in, &min,&max);

		for(int i=0;i<= max; i++){
			cv::Vec3b color(std::rand() % 255,std::rand() % 255,std::rand() % 255);
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

	void extract_results(results& pixel, results& Regions){
		pixel.precision = 100*pixel_precision;
		pixel.recall    = 100*pixel_recall;

		float cum_precision=0;
		float cum_recall=0;
		int size_precision=0, size_recall=0;

		for(int j=0; j < static_cast<int>(Precisions.back().size());j++){
			cum_precision += Precisions.back()[j];
			size_precision++;
		}
		for(int j=0; j < static_cast<int>(Recalls.back().size());j++){
			cum_recall    += Recalls.back()[j];
			size_recall++;
		}
		(void)size_precision;
		(void)size_recall;
		Regions.precision = cum_precision/Precisions.back().size();
		Regions.recall    = cum_recall/Recalls.back().size();
	}

	void process_all_files(){
		std::set<std::string> files_to_read = listFile();

		for (std::set<std::string>::iterator file_iter = files_to_read.begin() ; file_iter != files_to_read.end() ; file_iter++){
			RCLCPP_INFO(this->get_logger(), "Reading file '%s'", file_iter->c_str());
			process_files(*file_iter);
			publish_Image();
		}

		float cum_precision=0, cum_quad_precision =0;
		float cum_recall=0, cum_quad_recall=0;
		int size_precision=0, size_recall=0;
		for(int i=0; i < static_cast<int>(Precisions.size());i++){
			for(int j=0; j < static_cast<int>(Precisions[i].size());j++){
				cum_precision += Precisions[i][j];
				cum_quad_precision += Precisions[i][j]*Precisions[i][j];
				size_precision++;
			}
		}
		for(int i=0; i < static_cast<int>(Recalls.size());i++){
			for(int j=0; j < static_cast<int>(Recalls[i].size());j++){
				cum_recall    += Recalls[i][j];
				cum_quad_recall    += Recalls[i][j]*Recalls[i][j];
				size_recall++;
			}
		}
		double this_precision = cum_precision/size_precision;
		double this_recall    = cum_recall/size_recall;

		double this_quad_precision = cum_quad_precision/size_precision;
		double this_quad_recall    = cum_quad_recall/size_recall;

		double std_precision = std::sqrt(this_quad_precision - this_precision*this_precision);
		double std_recall = std::sqrt(this_quad_recall - this_recall*this_recall);

		RCLCPP_INFO(
			this->get_logger(),
			"Aggregated precision=%.1f +/- %.1f recall=%.1f +/- %.1f",
			this_precision,
			std_precision,
			this_recall,
			std_recall);

		cum_precision=0, cum_quad_precision =0;
		cum_recall=0, cum_quad_recall=0;
		for(int i=0; i < static_cast<int>(Precisions.size());i++){
			float cum_inside_precision = 0;
			for(int j=0; j < static_cast<int>(Precisions[i].size());j++){
				cum_inside_precision += Precisions[i][j];
			}
			cum_precision      += cum_inside_precision/Precisions[i].size();
			RCLCPP_DEBUG(
				this->get_logger(),
				"Per-map precision[%d]=%.1f",
				i,
				cum_inside_precision/Precisions[i].size());
			cum_quad_precision += cum_inside_precision/Precisions[i].size()*cum_inside_precision/Precisions[i].size();
		}
		size_precision=Precisions.size();

		for(int i=0; i < static_cast<int>(Recalls.size());i++){
			float cum_inside_recall = 0;
			for(int j=0; j < static_cast<int>(Recalls[i].size());j++){
				cum_inside_recall    += Recalls[i][j];
			}
			cum_recall += cum_inside_recall/Recalls[i].size();
			cum_quad_recall += cum_inside_recall/Recalls[i].size()*cum_inside_recall/Recalls[i].size();
			RCLCPP_DEBUG(
				this->get_logger(),
				"Per-map recall[%d]=%.1f",
				i,
				cum_inside_recall/Recalls[i].size());
		}
		size_recall=Recalls.size();

		this_precision = cum_precision/size_precision;
		this_recall    = cum_recall/size_recall;

		this_quad_precision = cum_quad_precision/size_precision;
		this_quad_recall    = cum_quad_recall/size_recall;

		std_precision = std::sqrt(this_quad_precision - this_precision*this_precision);
		std_recall = std::sqrt(this_quad_recall - this_recall*this_recall);

		RCLCPP_INFO(
			this->get_logger(),
			"Separated precision=%.1f +/- %.1f recall=%.1f +/- %.1f",
			this_precision,
			std_precision,
			this_recall,
			std_recall);
	}

	void read_all_files(){
		std::set<std::string> files_to_read = listFile();

		for (std::set<std::string>::iterator file_iter = files_to_read.begin() ; file_iter != files_to_read.end() ; file_iter++){
			cv::Mat image_GT;
			std::string full_path_GT = base_path + "/" + *file_iter + gt_ending;
			image_GT = cv::imread(full_path_GT,0);

			float rows = image_GT.rows;
			float cols = image_GT.cols;

			RCLCPP_INFO(
				this->get_logger(),
				"Reading file '%s' size_m=%.2fx%.2f",
				file_iter->c_str(),
				rows*0.05,
				cols*0.05);
		}
	}
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);

	float decomp_th=2.7;
	if (argc ==2){ decomp_th = std::atof(argv[1]); }

	auto node = std::make_shared<ROS_handler>(decomp_th);
	node->init_publishers();
	rclcpp::spin(node);
	rclcpp::shutdown();

	return 0;
}
