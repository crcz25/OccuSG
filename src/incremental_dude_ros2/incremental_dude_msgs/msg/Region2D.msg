# Region2D message for DuDe topological segmentation.
#
# IMPORTANT:
# - Raw DuDe detections use frame-local IDs that may reshuffle between frames.
# - `/dude/regions` from incremental_dude_ros2 may replace `id` with a
#   node-lifetime canonical tracker ID.
# - Long-term semantic room identity and split / merge lineage are NOT encoded here.

std_msgs/Header header

# Detection ID. Frame-local in raw detections; canonical on tracked output.
int32 id

# Region centroid in map frame (meters)
geometry_msgs/Pose2D centroid

# Region boundary polygon in map frame (meters)
# Points ordered counter-clockwise around region perimeter
geometry_msgs/Polygon polygon

# Convex hull of the region in map frame (meters)
geometry_msgs/Polygon convex_hull

# Region area in square meters
float32 area

# IDs of adjacent detections in the same message, using the same ID space as `id`.
int32[] adjacent_ids
