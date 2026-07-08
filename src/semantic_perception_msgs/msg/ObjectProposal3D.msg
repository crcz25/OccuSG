uint32 id
string class_name
uint32 class_id

vision_msgs/BoundingBox2D bbox_2d
sensor_msgs/Image mask

float32[] mask_embedding
float32[] bbox_embedding
float32[] fused_embedding

float32 detection_confidence
string detector_source

bool valid_3d
geometry_msgs/Point centroid_3d
vision_msgs/BoundingBox3D bbox_3d

# Reserved for a later semantic classification / uncertainty module.
float32 similarity_score
float32 entropy_score
