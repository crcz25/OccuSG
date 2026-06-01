# incremental_dude_ros2

ROS 2 port of the original ROS 1 `inc_dude` package in
`/workspace/occusg_ws/Incremental_DuDe_ROS`.

Current node surface:

- Subscribes to `map` (`nav_msgs/msg/OccupancyGrid`)
- Subscribes to `chatter` (`std_msgs/msg/String`) for snapshot saving
- Publishes `/tagged_image`
- Publishes `/dude/regions` (`incremental_dude_msgs/msg/Region2DArray`)

The ROS 2 port is intended to match the ROS 1 package behavior as closely as
possible. DuDe still produces frame-local region detections internally, but the
default `/dude/regions` output now passes those detections through a lightweight
post-processing tracker so `Region2D.id` is a node-lifetime canonical ID.

The region tracker is intentionally simple: it greedily matches current regions
to recent tracks using map-frame bounding-box IoU, centroid distance, and area
ratio gates. It preserves the rest of each region message and remaps
`adjacent_ids` from frame-local IDs to canonical IDs when publishing.

Tracker parameters live in `config/inc_dude_params.yaml`:

- `region_tracker_enable` enables canonical IDs on `/dude/regions`.
- `region_tracker_min_iou` accepts overlap matches above this bounding-box IoU.
- `region_tracker_max_centroid_distance` accepts nearby regions within this many meters.
- `region_tracker_min_area_ratio` and `region_tracker_max_area_ratio` gate size changes.
- `region_tracker_max_missed_frames` keeps disappeared tracks alive briefly.
- `region_tracker_publish_debug` emits throttled tracker stats.
