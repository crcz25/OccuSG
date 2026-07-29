uint32 id
string class_name
uint32 class_id

vision_msgs/BoundingBox2D bbox_2d

# Unit-norm CLIP components, each of dimension D.
#   mask_embedding  : CLIP image embedding of the SAM-masked crop (black background)
#   bbox_embedding  : CLIP image embedding of the unmasked detector-box crop
#   label_embedding : cached CLIP text embedding of class_name
float32[] mask_embedding
float32[] bbox_embedding
float32[] label_embedding

# concat(mask_embedding, bbox_embedding, label_embedding), L2-normalized.
# Dimension is exactly 3 * D.
float32[] fused_embedding

float32 detection_confidence
string detector_source

bool valid_3d
geometry_msgs/Point centroid_3d
vision_msgs/BoundingBox3D bbox_3d
