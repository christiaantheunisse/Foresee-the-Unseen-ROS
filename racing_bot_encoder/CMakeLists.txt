cmake_minimum_required(VERSION 3.8)
project(racing_bot_encoder)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(racing_bot_interfaces REQUIRED)

find_library(PIGPIO_IF2_LIB pigpiod_if2)

include_directories(include)

add_executable(encoder_node src/main.cpp src/encoder_node.cpp src/encoder_sensor.cpp)
ament_target_dependencies(encoder_node rclcpp racing_bot_interfaces)
target_link_libraries(encoder_node ${PIGPIO_IF2_LIB})

install(
  TARGETS
  encoder_node 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_encoder_test test/encoder_test.cpp src/encoder_node.cpp src/encoder_sensor.cpp)
  target_include_directories(${PROJECT_NAME}_encoder_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${PROJECT_NAME}_encoder_test
    rclcpp std_msgs
  )
  target_link_libraries(${PROJECT_NAME}_encoder_test ${PIGPIO_IF2_LIB})
endif()

ament_package()