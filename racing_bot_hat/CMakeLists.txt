cmake_minimum_required(VERSION 3.8)
project(racing_bot_hat)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

find_library(PIGPIO_IF2_LIB pigpiod_if2)

include_directories(include)

add_executable(hat_node src/main.cpp src/hat_node.cpp)
ament_target_dependencies(hat_node rclcpp std_msgs)
target_link_libraries(hat_node ${PIGPIO_IF2_LIB})

install(
  TARGETS
  hat_node 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_hat_test test/hat_test.cpp src/hat_node.cpp)
  target_include_directories(${PROJECT_NAME}_hat_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${PROJECT_NAME}_hat_test
    rclcpp rclcpp std_msgs
  )
  target_link_libraries(${PROJECT_NAME}_hat_test ${PIGPIO_IF2_LIB})
endif()

ament_package()
