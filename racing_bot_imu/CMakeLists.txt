cmake_minimum_required(VERSION 3.8)
project(racing_bot_imu)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_library(WIRINGPI_LIB wiringPi)
find_package(sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)

include_directories(${sensor_msgs_INCLUDE_DIRS})

set(LSM9DS1_SRCS
  include/LSM9DS1_RaspberryPi_Library/include/LSM9DS1.h
  include/LSM9DS1_RaspberryPi_Library/src/LSM9DS1.cpp
)


add_library(LSM9DS1 STATIC ${LSM9DS1_SRCS})


include_directories(include)

add_executable(imu_node src/main.cpp src/imu_node.cpp src/MadgwickAHRS.cpp)
ament_target_dependencies(imu_node rclcpp sensor_msgs tf2 tf2_geometry_msgs)
target_link_libraries(imu_node LSM9DS1 ${WIRINGPI_LIB})



install(
  TARGETS
  imu_node 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_imu_test test/imu_test.cpp src/imu_node.cpp src/MadgwickAHRS.cpp)
  target_include_directories(${PROJECT_NAME}_imu_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${PROJECT_NAME}_imu_test
    rclcpp sensor_msgs tf2 tf2_geometry_msgs
  )
  target_link_libraries(${PROJECT_NAME}_imu_test LSM9DS1 ${WIRINGPI_LIB})
endif()

ament_package()

