cmake_minimum_required(VERSION 3.8)
project(semantic_perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectProposal3D.msg"
  "msg/ObjectProposal3DArray.msg"
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
