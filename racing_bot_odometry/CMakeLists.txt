cmake_minimum_required(VERSION 3.8)
project(racing_bot_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(racing_bot_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(include)

add_executable(odometry_node src/main.cpp src/odometry_node.cpp)
ament_target_dependencies(odometry_node rclcpp std_msgs nav_msgs tf2 tf2_geometry_msgs tf2_ros racing_bot_interfaces)

install(
  TARGETS 
  odometry_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()