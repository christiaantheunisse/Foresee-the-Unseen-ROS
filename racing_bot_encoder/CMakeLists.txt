cmake_minimum_required(VERSION 3.8)
project(racing_bot_encoder)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(racing_bot_interfaces REQUIRED)

find_library(PIGPIO_IF2_LIB pigpiod_if2)

include_directories(include)

add_executable(encoder_node src/main.cpp src/encoder_node.cpp src/encoder_sensor.cpp)
ament_target_dependencies(encoder_node rclcpp std_msgs racing_bot_interfaces)
target_link_libraries(encoder_node ${PIGPIO_IF2_LIB})

install(
  TARGETS
  encoder_node 
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()