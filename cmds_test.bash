source /opt/ros/humble/setup.bash
source /home/devuser/venv/bin/activate
cd /workspace/occusg_ws
source install/setup.bash

ros2 run semantic_perception semantic_perception_node --ros-args \
  --params-file src/semantic_perception/config/semantic_perception.yaml \
  -p device:=cpu \
  -p devices:="['cpu']" \
  -p num_worker_threads:=1 \
  -p frame_queue_size:=1 \
  -r /camera/color/image_raw:=/rgb \
  -r /camera/depth/image_raw:=/depth \
  -r /camera/color/camera_info:=/rgb/camera_info \
  -r image:=/semantic_perception/debug_image


