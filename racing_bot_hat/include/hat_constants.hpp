#ifndef HATCONSTANTS_H
#define HATCONSTANTS_H

#include <string>

namespace racing_bot
{
  namespace hat
  {

    const std::string SUBSCRIPTION_TOPIC = "cmd_motor";
    const std::string NODE_NAME = "hat_node";
    const int MOTOR_QUEUE_SIZE = 5;
  }
}
#endif
