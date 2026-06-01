#include "wrapper.hpp"
#include "dude_ros_logging.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <utility>

extern Draw_Decoration draw_decoration;

namespace {

constexpr double kMinPolygonAreaPixels = 1.0;

bool isValidContourForPolygon(const std::vector<cv::Point> &contour,
                              const double min_area = kMinPolygonAreaPixels) {
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

std::vector<cv::Point> orientedForDuDe(
	std::vector<cv::Point> contour,
	const c_ply::POLYTYPE polygon_type) {
	const double signed_area = cv::contourArea(contour, true);
	const bool want_opencv_positive = polygon_type == c_ply::PIN;
	if ((want_opencv_positive && signed_area < 0.0) ||
	    (!want_opencv_positive && signed_area > 0.0)) {
		std::reverse(contour.begin(), contour.end());
	}
	return contour;
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

void resetThirdPartyGlobalDecoration() {
	::draw_decoration.se.destroy();
	::draw_decoration.tvec.clear();
	::draw_decoration.VIP_vertices.clear();
	::draw_decoration.allAccumulatedCuts.clear();
	::draw_decoration.allFeaturePMs.clear();
	::draw_decoration.allFeatureConcavities.clear();
	::draw_decoration.repDiags.clear();
	::draw_decoration.userDiags.clear();
	::draw_decoration.compDiags.clear();
	::draw_decoration.holeCutTreeLines.clear();
	::draw_decoration.MAXCONCAVITY = -FLT_MAX;
	::draw_decoration.MINCONCAVITY = FLT_MAX;
	::draw_decoration.current_imgID = 0;
	::draw_decoration.R = 0;
	::draw_decoration.box[0] = FLT_MAX;
	::draw_decoration.box[1] = -FLT_MAX;
	::draw_decoration.box[2] = FLT_MAX;
	::draw_decoration.box[3] = -FLT_MAX;
	::draw_decoration.normal_length = 1;
	::draw_decoration.selectedVIPVertex = -1;
	::draw_decoration.g_selected_PM = NULL;
}

class ScopedThirdPartyDecorationReset {
public:
	ScopedThirdPartyDecorationReset() {
		resetThirdPartyGlobalDecoration();
	}

	~ScopedThirdPartyDecorationReset() {
		resetThirdPartyGlobalDecoration();
	}
};

}  // namespace

DuDe_OpenCV_wrapper::DuDe_OpenCV_wrapper(){
    Area_threshold=5; 
    dude= c_dude();
	tolerance_in_pixels= false;	    
	g_tau= 0.2;
    g_tau_pixel=60; //60 pixels
}

DuDe_OpenCV_wrapper::~DuDe_OpenCV_wrapper(){
	std::unordered_set<c_polygon*> released_polygons;
	auto release_polygon = [&released_polygons](c_polygon* polygon_ptr) {
		if (polygon_ptr == nullptr || !released_polygons.insert(polygon_ptr).second) {
			return;
		}
		polygon_ptr->destroy();
		delete polygon_ptr;
	};

	for(size_t i=0; i < finalPolygonPieces.size(); i++){
		release_polygon(finalPolygonPieces[i]);
	}
	finalPolygonPieces.clear();

	for(size_t i=0; i < owned_polygons_.size(); i++){
		release_polygon(owned_polygons_[i]);
	}
	owned_polygons_.clear();
//	dude.clear();
	
	for(size_t i=0; i < 	draw_decoration.allFeaturePMs.size(); i++){
		ply_vertex* current_ply = 	draw_decoration.allFeaturePMs[i];
		(void)current_ply;
	//		current_ply->destroy();
	}
	draw_decoration.allFeaturePMs.clear();
		
	for(size_t i=0; i < draw_decoration.allAccumulatedCuts.size(); i++){
		c_diagonal current_diagonal = draw_decoration.allAccumulatedCuts[i];
		(void)current_diagonal;
	//		current_diagonal->destroy();
	}	
	draw_decoration.allAccumulatedCuts.clear();	

	draw_decoration.se.destroy();
}




///////////////////////////////////
void DuDe_OpenCV_wrapper::insert_contour_to_poly(std::vector<cv::Point> contour_in, c_ply& polygon ){

	if (!isValidContourForPolygon(contour_in)) {
		return;
	}
	contour_in = orientedForDuDe(std::move(contour_in), polygon.getType());
	polygon.beginPoly();
	for(int i=1;i <= static_cast<int>(contour_in.size());i++){
		float x = contour_in[contour_in.size()-i].x;
		float y = contour_in[contour_in.size()-i].y;
		polygon.addVertex(x, y);
	}
	polygon.endPoly();
	
}	


////////////////////////////////////////
cv::Rect DuDe_OpenCV_wrapper::Decomposer(cv::Mat Occ_Image){
	ScopedThirdPartyDecorationReset third_party_decoration_scope;
	Parent_contour.clear();
	Decomposed_contours.clear();
	contours_centroid.clear();
	contours_connections.clear();
	diagonal_centroid.clear();
	diagonal_connections.clear();
	last_input_contours_ = 0;
	last_valid_polygons_ = 0;
	last_rejected_polygons_ = 0;
	last_cuts_after_resolution_ = 0;

	if (Occ_Image.empty() || Occ_Image.rows == 0 || Occ_Image.cols == 0) {
		return cv::Rect();
	}
///////////////////////////////	
//// Look for the big contours
	double start_finding = getTime();
	std::vector<std::vector<cv::Point> > Explored_contour;
	std::vector<cv::Vec4i> hierarchy; //[Next, Previous, First_Child, Parent]
	cv::findContours(Occ_Image, Explored_contour, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
	last_input_contours_ = Explored_contour.size();
	DUDE_ROS_DEBUGF_THROTTLE(
		2000,
		"wrapper input_image=%dx%d extracted_contours=%zu hierarchy=%zu",
		Occ_Image.cols,
		Occ_Image.rows,
		Explored_contour.size(),
		hierarchy.size());
	if (Explored_contour.empty() || hierarchy.empty()) {
		return cv::Rect();
	}
		
	cv::Rect resize_rect;
	double Big_Contour_area = 0.0;
	int Big_Contour_Index = -1;
	for (int contour_index = 0; contour_index < static_cast<int>(Explored_contour.size()); ++contour_index) {
		if (hierarchy[contour_index][3] != -1) {
			continue;
		}
		if (!isValidContourForPolygon(Explored_contour[contour_index], Area_threshold)) {
			++last_rejected_polygons_;
			continue;
		}
		const double current_area = std::abs(cv::contourArea(Explored_contour[contour_index]));
		const cv::Rect current_rect = boundingRect(Explored_contour[contour_index]);
		resize_rect = resize_rect.area() > 0 ? (resize_rect | current_rect) : current_rect;
		if (Big_Contour_Index < 0 || current_area > Big_Contour_area) {
			Big_Contour_area = current_area;
			Big_Contour_Index = contour_index;
		}
	}
	if (Big_Contour_Index < 0) {
		DUDE_ROS_WARNF(
			"wrapper rejected frame reason=no_valid_outer_polygon input_contours=%zu rejected=%zu",
			last_input_contours_,
			last_rejected_polygons_);
		return cv::Rect();
	}
	double end_finding = getTime();
	Parent_contour = Explored_contour [Big_Contour_Index ];
	int inserted_holes = 0;
/////////////////////////////
//// Insert contours in Dual-Space Decomposer 
	double start_inserting = getTime();
	
	c_polygon polygons;
	{
		c_ply poly_out(c_ply::POUT);
		insert_contour_to_poly( Explored_contour [Big_Contour_Index ], poly_out);
		if (poly_out.getHead() == nullptr || poly_out.getSize() < 3) {
			DUDE_ROS_WARNF(
				"wrapper rejected frame reason=invalid_outer_polygon input_contours=%zu rejected=%zu",
				last_input_contours_,
				last_rejected_polygons_ + 1);
			return cv::Rect();
		}
		polygons.push_back(poly_out);
		++last_valid_polygons_;

		int current_child = hierarchy[ Big_Contour_Index ] [2];				
		while(current_child !=-1){ //if child, insert
			const double Area = std::abs(cv::contourArea(Explored_contour[current_child]));
			if(Area > Area_threshold && isValidContourForPolygon(Explored_contour[current_child], Area_threshold)){
				c_ply poly_in(c_ply::PIN);
				insert_contour_to_poly( Explored_contour [current_child ], poly_in);
				if (poly_in.getHead() != nullptr && poly_in.getSize() >= 3) {
					polygons.push_back(poly_in);
					++inserted_holes;
					++last_valid_polygons_;
				} else {
					++last_rejected_polygons_;
				}
			} else if (Area > Area_threshold) {
				++last_rejected_polygons_;
			}
			current_child = hierarchy[ current_child ] [0];
		}
	}
	double end_inserting = getTime();
	DUDE_ROS_DEBUGF_THROTTLE(
		2000,
		"wrapper polygons_to_decompose outer_area=%.1f holes=%d valid_polygons=%zu rejected_polygons=%zu find_ms=%.1f insert_ms=%.1f",
		Big_Contour_area,
		inserted_holes,
		last_valid_polygons_,
		last_rejected_polygons_,
		end_finding - start_finding,
		end_inserting - start_inserting);
	
////////////////////////////////
//// Decompose
	if (polygons.empty() || polygons.front().getHead() == nullptr || polygons.front().getSize() < 3) {
		DUDE_ROS_WARNF(
			"wrapper rejected frame reason=no_polygons_to_decompose input_contours=%zu valid_polygons=%zu rejected_polygons=%zu",
			last_input_contours_,
			last_valid_polygons_,
			last_rejected_polygons_);
		return cv::Rect();
	}
	float r = polygons.front().getRadius();

	if(tolerance_in_pixels){
		 g_tau = g_tau_pixel;
	 }
	 else{
		 g_tau*=r;// Scale the concavity variable if tau is fractional
	 }
	double start_dude = getTime();
	///////////////
	Dual_decompose(polygons);
	//////////////
	double end_dude = getTime();
	DUDE_ROS_DEBUGF_THROTTLE(
		2000,
		"wrapper raw_decomposed_polygons=%zu tau=%.3f cuts_after_resolution=%zu decompose_ms=%.1f",
		finalPolygonPieces.size(),
		g_tau,
		last_cuts_after_resolution_,
		end_dude - start_dude);

///////////////////////
// Polygon to Contour
	extract_contour_from_polygon();		
	DUDE_ROS_DEBUGF_THROTTLE(
		2000,
		"wrapper raw_decomposed_regions=%zu centroids=%zu valid_polygons=%zu rejected_polygons=%zu resize_rect=(%d,%d,%d,%d)",
		Decomposed_contours.size(),
		contours_centroid.size(),
		last_valid_polygons_,
		last_rejected_polygons_,
		resize_rect.x,
		resize_rect.y,
		resize_rect.width,
		resize_rect.height);

	return resize_rect;
}

//////////////////////////////////////////////////////////////
void DuDe_OpenCV_wrapper::Dual_decompose(c_polygon& polygons){
	c_polygon* p1 = new c_polygon();	
	owned_polygons_.push_back(p1);
	p1->copy(polygons);
	dude.build(*p1, g_tau, true);
	last_cuts_after_resolution_ = dude.getFinalCuts().size();
	getP()=polygons;

	prepare_skeleton();
	
	draw_decoration.allFeaturePMs.insert(draw_decoration.allFeaturePMs.end(), dude.m_PMs.begin(), dude.m_PMs.end());
	draw_decoration.allFeaturePMs.insert(draw_decoration.allFeaturePMs.end(), dude.hole_PMs.begin(), dude.hole_PMs.end());
	updateMINMAXConcavity(dude);
	
	//debug
	polygons.build_all();

	iterativeDecompose(*p1, dude.m_cuts, g_tau, finalPolygonPieces, draw_decoration.allAccumulatedCuts, draw_decoration.se, true);
//	decomposeMoreTimes(polygons, dude.m_cuts, g_tau, finalPolygonPieces, draw_decoration.allAccumulatedCuts, draw_decoration.se, true, 0);

	estimate_COM_R_Box(); //compute the bounding box of all geometries

}

//////////////////////
void DuDe_OpenCV_wrapper::prepare_skeleton()
{
	//prepare for extract skeletons
	ExtracSkeleton * m_ES = ES_Factory::create_ES("PA");
	draw_decoration.se.setExtracSkeleton(m_ES);
	draw_decoration.se.setQualityMeasure(NULL);
	draw_decoration.se.begin();
}

//////////////////////////////
void DuDe_OpenCV_wrapper::export_all_svg_files()
{
	string basename = "Decomposed";
	int psize = finalPolygonPieces.size();

	c_polygon & P = getP();
	double * bbox = P.getBBox();
	double width = bbox[1] - bbox[0];
	double height = bbox[3] - bbox[2];
	double stroke_width = width / 200;
	svg::Dimensions dimensions(800, 800);
	svg::Dimensions svg_viewbox_dim(width + stroke_width * 2, height + stroke_width * 2);
	svg::Point svg_viewbox_org(bbox[0] - stroke_width, height - bbox[3] + stroke_width);

	stringstream ss;
	ss << basename.c_str() << "_dude2d.svg";
	string sname = ss.str();
	svg::Document doc(sname, svg::Layout(dimensions, svg::Layout::BottomLeft, svg_viewbox_dim, svg_viewbox_org));


	for (int i = 0; i < psize; i++)
	{
		svg::Color randColor(svg::Color::Defaults(svg::Color::Aqua + (i % 15)));
		svg::Polygon p_i(svg::Fill(randColor), svg::Stroke(stroke_width, svg::Color::Black));
		finalPolygonPieces[i]->toSVG(p_i);
		doc << p_i;
	}

	doc.save();
	DUDE_ROS_INFOF("wrapper saved_svg path=%s polygons=%d", sname.c_str(), psize);
}

/////////////////////////////////////
void DuDe_OpenCV_wrapper::estimate_COM_R_Box() 
{
	getP().buildBoxAndCenter();
	getQ().buildBoxAndCenter();

	for (short i = 0; i < 4; i++)
		draw_decoration.box[i] = getP().getBBox()[i]; //+getQ().getBBox()[i];

	draw_decoration.COM.set((draw_decoration.box[1] + draw_decoration.box[0]) / 2, (draw_decoration.box[3] + draw_decoration.box[2]) / 2);
	draw_decoration.R = sqr(draw_decoration.COM[0]-draw_decoration.box[0]) + sqr(draw_decoration.COM[1] - draw_decoration.box[2]);
	draw_decoration.R = sqrt((float) draw_decoration.R); //*0.95;

	//rebuild box
	draw_decoration.box[0] = draw_decoration.COM[0] - draw_decoration.R;
	draw_decoration.box[1] = draw_decoration.COM[0] + draw_decoration.R;
	draw_decoration.box[2] = draw_decoration.COM[1] - draw_decoration.R;
	draw_decoration.box[3] = draw_decoration.COM[1] + draw_decoration.R;

	//estimate normal length based on R
	draw_decoration.normal_length = draw_decoration.R / 20;
}

/////////////////////////////
void DuDe_OpenCV_wrapper::extract_contour_from_polygon(){

	Decomposed_contours.clear();
	contours_centroid.clear();
	for(size_t i=0; i < finalPolygonPieces.size(); i++){
		if (finalPolygonPieces[i] == nullptr) {
			++last_rejected_polygons_;
			continue;
		}
		c_polygon temp_polygon = *finalPolygonPieces[i];
		if (temp_polygon.begin() == temp_polygon.end()) {
			++last_rejected_polygons_;
			continue;
		}
		list<c_ply>::iterator temp_iter = temp_polygon.begin();
		
		std::vector<cv::Point> current_contour;

		ply_vertex * ptr = temp_iter->getHead(); // There's only one polygon after decomposed
		if (ptr == nullptr) {
			++last_rejected_polygons_;
			continue;
		}
		do{
			const Point2d& pos = ptr->getPos();
			cv::Point point_to_add(pos[0], pos[1]);
			current_contour.push_back(point_to_add);
			ptr = ptr->getNext();
		} 
		while (ptr != nullptr && ptr!=temp_iter->getHead());
		if (!isValidContourForPolygon(current_contour)) {
			++last_rejected_polygons_;
			continue;
		}
		Decomposed_contours.push_back(current_contour);
		contours_centroid.push_back(contourCentroidOrFirst(current_contour));

	}
}

///////////////////////////////////////
void DuDe_OpenCV_wrapper::extract_graph(){
	//////////////Diagonals are the edges and contours are the nodes
	vector<c_diagonal> Diagonals = dude.getFinalCuts();		
	contours_connections.resize(Decomposed_contours.size());

	////// Find diagonals in contours	
	int diag_count=0;
	(void)diag_count;
	diagonal_centroid.clear();
	diagonal_connections.clear();
	
	for (vector<c_diagonal>::iterator Diag_it = Diagonals.begin(); Diag_it != Diagonals.end(); ++Diag_it){	
		std::set<int> contours_connected;
		ply_vertex * vertex1 = Diag_it->getV1();
		const Point2d& pos_vertex1 = vertex1->getPos();
		ply_vertex * vertex2 = Diag_it->getV2();
		const Point2d& pos_vertex2 = vertex2->getPos();

		cv::Point Diag_avg = cv::Point( (pos_vertex1[0] + pos_vertex2[0])/2, (pos_vertex1[1] + pos_vertex2[1])/2  );
		diagonal_centroid.push_back(Diag_avg);

		for(int contour_count = 0; contour_count < static_cast<int>(Decomposed_contours.size()); contour_count++){
			std::vector<cv::Point> current_contour = Decomposed_contours[contour_count];

			if( abs( cv::pointPolygonTest(current_contour, Diag_avg, true) ) <= 2){
				contours_connected.insert(contour_count);
			}
		}
		diagonal_connections.push_back(contours_connected);
		
		for(std::set<int>::iterator  set_it = contours_connected.begin();set_it!=contours_connected.end();set_it++){
			int connected_from = *set_it;
			for(std::set<int>::iterator  set_it2 = contours_connected.begin();set_it2!=contours_connected.end();set_it2++){
				int connected_to = *set_it2;
				if(connected_from != connected_to){
					contours_connections[connected_from].insert(connected_to);
				}
			}
		}
	// */
	}
}

////////////////////////////	
void DuDe_OpenCV_wrapper::print_graph(){
	DUDE_ROS_DEBUG_STREAM() << "wrapper graph nodes=" << contours_centroid.size();
	for(size_t i=0;i<contours_centroid.size();i++){
		std::ostringstream node_stream;
		node_stream << "node " << i << " connected_to=(";
		for(std::set<int>::iterator  set_it = contours_connections[i].begin();set_it!=contours_connections[i].end();set_it++){
			int connected_from = *set_it;
			node_stream << " " << connected_from;
		}
		node_stream << " )";
		DUDE_ROS_DEBUG_STREAM() << node_stream.str();
	}
}


///////////////////////////
void DuDe_OpenCV_wrapper::measure_performance(){
	vector<float> convexities, compactness, qualities, areas;
	float Total_Area=0;
	float Total_Quality=0;

	for (int i=0; i < static_cast<int>(Decomposed_contours.size()); i++){

		cv::Moments moments = cv::moments(Decomposed_contours[i], true);
		vector<cv::Point> hull;
		
		convexHull( Decomposed_contours[i], hull );	
		float A_i=moments.m00;
		float H_i = cv::contourArea(hull);
		float c_i = A_i/H_i;
		convexities.push_back( c_i);
		areas.push_back( A_i);
		float M_i=A_i;// currently the number of cells is the number of pixels

		float s_i = moments.mu20 + moments.mu02;
		s_i = (1/(M_i*A_i))*(s_i/A_i);
		compactness.push_back(s_i);
		
		float q_i = c_i - s_i;
		qualities.push_back(q_i);
		
		Total_Area+=A_i;
		Total_Quality+=q_i;
	}
	
	float Parent_Area = cv::contourArea(Parent_contour);
	float Area_Coverage_Ratio =(Total_Area/Parent_Area);//Regularly 1
	float Validity_Ratio = 1;//All areas are valid
	float Simplicity=1; // regions coincide with user defined

	float Overall_Quality;
	float lambda=1; //equally weighted

	Overall_Quality = (Area_Coverage_Ratio * Validity_Ratio)/(contours_centroid.size())*Total_Quality + lambda*Simplicity;
	DUDE_ROS_DEBUGF(
		"wrapper quality coverage=%.3f validity=%.3f regions=%zu simplicity=%.3f avg_quality=%.3f overall=%.3f",
		Area_Coverage_Ratio,
		Validity_Ratio,
		contours_centroid.size(),
		Simplicity,
		Total_Quality/contours_centroid.size(),
		Overall_Quality);
}
