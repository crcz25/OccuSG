cmake_minimum_required(VERSION 3.8)
project(semantic_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common features filters io segmentation search)
find_package(pcl_conversions REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Show full compiler/linker commands" FORCE)

# ONNX Runtime configuration
option(ONNXRUNTIME_USE_GPU "Enable the CUDA provider for ONNX Runtime" OFF)
set(ONNXRUNTIME_CPU_DEFAULT_DIR "/opt/onnxruntime-linux-x64-1.19.2")
set(ONNXRUNTIME_GPU_DEFAULT_DIR "/opt/onnxruntime-linux-x64-gpu-1.19.2")
if(ONNXRUNTIME_USE_GPU)
  set(_onnxruntime_default_dir "${ONNXRUNTIME_GPU_DEFAULT_DIR}")
else()
  set(_onnxruntime_default_dir "${ONNXRUNTIME_CPU_DEFAULT_DIR}")
endif()
set(ONNXRUNTIME_DIR "" CACHE PATH "Root dir of ONNX Runtime (e.g. /opt/onnxruntime-linux-x64-1.19.2)")

# Keep ONNXRUNTIME_DIR auto-selectable when using package defaults, so toggling
# ONNXRUNTIME_USE_GPU switches between CPU/GPU runtime trees deterministically.
if(ONNXRUNTIME_DIR STREQUAL "" OR
   ONNXRUNTIME_DIR STREQUAL "${ONNXRUNTIME_CPU_DEFAULT_DIR}" OR
   ONNXRUNTIME_DIR STREQUAL "${ONNXRUNTIME_GPU_DEFAULT_DIR}")
  set(ONNXRUNTIME_DIR "${_onnxruntime_default_dir}" CACHE PATH
      "Root dir of ONNX Runtime (e.g. /opt/onnxruntime-linux-x64-1.19.2)" FORCE)
endif()

message(STATUS "ONNXRUNTIME_USE_GPU: ${ONNXRUNTIME_USE_GPU}")
message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_DIR}")
if(NOT EXISTS "${ONNXRUNTIME_DIR}")
  message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_DIR}")
endif()

# locate the core ORT libs
# Re-resolve on every configure so ONNXRUNTIME_USE_GPU toggles don't keep a
# stale cached library path from a previous build mode.
unset(ORT_CORE_LIB CACHE)
find_library(ORT_CORE_LIB
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_DIR}/lib
  NO_DEFAULT_PATH
)
if(NOT ORT_CORE_LIB)
  message(FATAL_ERROR "Could not find onnxruntime library in ${ONNXRUNTIME_DIR}/lib")
endif()
message(STATUS "ONNX Runtime library: ${ORT_CORE_LIB}")

set(YOLOS_CPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/YOLOs-CPP/include)
if(NOT EXISTS "${YOLOS_CPP_INCLUDE_DIR}/yolos/tasks/segmentation.hpp")
  message(FATAL_ERROR "YOLOs-CPP headers not found at ${YOLOS_CPP_INCLUDE_DIR}")
endif()

add_executable(${PROJECT_NAME} src/semantic_node.cpp)

# compile-time flag for the code
if(ONNXRUNTIME_USE_GPU)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
  message(STATUS "ONNX Runtime with CUDA support enabled")
endif()

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${ONNXRUNTIME_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/YOLOs-CPP/include
)

target_compile_features(${PROJECT_NAME} PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17

target_link_libraries(${PROJECT_NAME}
  ${ORT_CORE_LIB}
  Threads::Threads
)

ament_target_dependencies(${PROJECT_NAME}
  cv_bridge
  OpenCV
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  message_filters
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  PCL
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
