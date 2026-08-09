source /opt/ros/humble/setup.bash
source /home/devuser/venv/bin/activate
cd /workspace/occusg_ws
source install/setup.bash

ros2 run semantic_perception semantic_perception_node --ros-args \
  --params-file src/semantic_perception/config/semantic_perception.yaml \
  -r /camera/color/image_raw:=/rgb \
  -r /camera/depth/image_raw:=/depth \
  -r /camera/color/camera_info:=/rgb/camera_info \
  -r image:=/semantic_perception/debug_image

ros2 launch semantic_perception semantic_perception.launch.py \
  config:=$PWD/src/semantic_perception/config/semantic_perception.yaml

ros2 launch scene_graph_ros scene_graph_pipeline_mp3d_bag.launch.py   bag_path:=$PWD/bags scan_id:=2t7WUuJeko7

source /opt/ros/humble/setup.bash
cd /workspace/occusg_ws
ros2 bag play bags/2t7WUuJeko7 \
  --rate 0.1 \
  --topics /rgb /depth /rgb/camera_info
