#ifndef IMUCONSTANTS_H
#define IMUCONSTANTS_H

#include <string>

namespace racing_bot
{
  namespace imu
  {

    const std::string PUBLISHER_TOPIC = "imu_data";
    const std::string NODE_NAME = "imu_node";
    const std::string FRAME_ID = "imu_link";
    const int IMU_QUEUE_SIZE = 10;
    const int TIMER_MILLISECONDS = 10;
    const int FILTER_SAMPLE_FREQUENCY = 100;
  }
}
#endif
