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
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(include)

add_executable(odometry_node src/main.cpp src/odometry_node.cpp)
ament_target_dependencies(odometry_node rclcpp std_msgs nav_msgs tf2 tf2_geometry_msgs tf2_ros)

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

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_odometry_test test/odometry_test.cpp src/odometry_node.cpp)
  target_include_directories(${PROJECT_NAME}_odometry_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${PROJECT_NAME}_odometry_test
    rclcpp std_msgs nav_msgs tf2 tf2_geometry_msgs tf2_ros
  )
  target_link_libraries(${PROJECT_NAME}_odometry_test)
endif()
ament_package()