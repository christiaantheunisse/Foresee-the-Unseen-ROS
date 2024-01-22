cmake_minimum_required(VERSION 3.8)
project(racing_bot_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(include)

add_executable(controller_node src/main.cpp src/controller_node.cpp)
ament_target_dependencies(controller_node rclcpp geometry_msgs std_msgs)

install(
  TARGETS
  controller_node 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_controller_test test/controller_test.cpp src/controller_node.cpp)
  target_include_directories(${PROJECT_NAME}_controller_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${PROJECT_NAME}_controller_test
    rclcpp geometry_msgs std_msgs
  )
  target_link_libraries(${PROJECT_NAME}_controller_test)
endif()

ament_package()
