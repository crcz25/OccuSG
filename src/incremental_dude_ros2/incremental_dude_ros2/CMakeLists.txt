cmake_minimum_required(VERSION 3.8)
project(incremental_dude_ros2)
include(CTest)

# Ensure <PackageName>_ROOT variables are honored by find_package.
cmake_policy(SET CMP0074 NEW)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set debug flags and disable assertions in DuDe library to prevent crashes
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DNDEBUG")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(incremental_dude_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# Force CGAL discovery to use the system install under /usr.
if(NOT DEFINED CGAL_ROOT)
  set(CGAL_ROOT "/usr")
endif()
if(NOT DEFINED CGAL_DIR)
  set(CGAL_DIR "/usr/lib/x86_64-linux-gnu/cmake/CGAL")
endif()
# Optional: provide CGAL_DATA_DIR to avoid noisy warning on distro packages
# that ship headers/libs but no bundled data folder. This project does not use
# CGAL::data_file_path(), so any existing path is acceptable as fallback.
if(NOT DEFINED CGAL_DATA_DIR OR CGAL_DATA_DIR STREQUAL "")
  if(EXISTS "/usr/share/CGAL/data")
    set(CGAL_DATA_DIR "/usr/share/CGAL/data")
  elseif(EXISTS "/usr/share/cgal")
    set(CGAL_DATA_DIR "/usr/share/cgal")
  else()
    set(CGAL_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endif()

# Find CGAL and MPFR (required for DuDe algorithm)
find_package(CGAL QUIET COMPONENTS Core)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(MPFR REQUIRED)

# Fallback for Ubuntu libcgal-dev config bug (CGAL_FOUND FALSE under /lib)
if(NOT CGAL_FOUND)
  set(CGAL_ROOT "/usr")
  set(CGAL_MODULES_DIR "/usr/lib/x86_64-linux-gnu/cmake/CGAL")
  list(APPEND CMAKE_MODULE_PATH "${CGAL_MODULES_DIR}")
  set(CGAL_INCLUDE_DIRS "/usr/include")
  set(CGAL_USE_FILE "${CGAL_MODULES_DIR}/UseCGAL.cmake")
  find_package(GMP REQUIRED)
  # MPFR already found above
  set(CGAL_LIBRARIES "")
  set(CGAL_3RD_PARTY_LIBRARIES ${GMP_LIBRARIES} ${MPFR_LIBRARIES})
endif()

# Include directories
include_directories(
  include
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/mathtool
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src/skeleton
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src/draw
  ${CGAL_INCLUDE_DIRS}
)
if(CGAL_USE_FILE)
  include(${CGAL_USE_FILE})
endif()

# Collect DuDe source files
file(GLOB DuDe_SOURCES
   ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src/*.cpp
   ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src/skeleton/*.cpp
   ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src/graph/*.cpp
   ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/src/draw/*.cpp
)

# Dependencies list for ament
set(dependencies
  rclcpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  incremental_dude_msgs
  cv_bridge
  image_transport
)

#############
## inc_dude node
#############
add_executable(inc_dude
  src/inc_dude.cpp
  include/wrapper.cpp
  include/inc_decomp.cpp
  ${DuDe_SOURCES}
)

ament_target_dependencies(inc_dude ${dependencies})

target_link_libraries(inc_dude
  ${OpenCV_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/FIST/libFIST.linux64.a
  ${CGAL_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  ${MPFR_LIBRARIES}
)

# Add link options to fix PIE linker error with static library
target_link_options(inc_dude PRIVATE "-no-pie")

#############
## evaluation node
#############
add_executable(evaluation
  src/evaluation.cpp
  include/wrapper.cpp
  include/inc_decomp.cpp
  ${DuDe_SOURCES}
)

ament_target_dependencies(evaluation ${dependencies})

target_link_libraries(evaluation
  ${OpenCV_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/FIST/libFIST.linux64.a
  ${CGAL_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  ${MPFR_LIBRARIES}
)

# Add link options to fix PIE linker error with static library
target_link_options(evaluation PRIVATE "-no-pie")

#############
## Install
#############

# Install executables
install(TARGETS
  inc_dude
  evaluation
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install maps directory
install(DIRECTORY
  maps
  DESTINATION share/${PROJECT_NAME}
)

# Install Third_Party directory (contains FIST library)
install(DIRECTORY
  Third_Party
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  add_executable(inc_decomp_regression_test
    test/inc_decomp_regression_test.cpp
    include/wrapper.cpp
    include/inc_decomp.cpp
    ${DuDe_SOURCES}
  )

  ament_target_dependencies(inc_decomp_regression_test ${dependencies})

  target_link_libraries(inc_decomp_regression_test
    ${OpenCV_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Third_Party/dude_final/FIST/libFIST.linux64.a
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    ${MPFR_LIBRARIES}
  )

  target_link_options(inc_decomp_regression_test PRIVATE "-no-pie")

  add_test(
    NAME inc_decomp_regression_test
    COMMAND inc_decomp_regression_test
  )

  add_executable(region_tracker_test
    test/region_tracker_test.cpp
  )

  ament_target_dependencies(region_tracker_test
    geometry_msgs
    incremental_dude_msgs
  )

  add_test(
    NAME region_tracker_test
    COMMAND region_tracker_test
  )
endif()

ament_package()
