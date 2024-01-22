#include <pigpiod_if2.h>
#include "encoder_sensor.hpp"

namespace racing_bot
{
  namespace encoder
  {
    EncoderSensor::EncoderSensor(const int pi_handle, const unsigned int pin_a, const unsigned int pin_b)
        : pi_handle_(pi_handle),
          pin_a_(pin_a),
          pin_b_(pin_b),
          position_(0),
          state_(0)

    {
      set_mode(pi_handle_, pin_a_, PI_INPUT);
      set_mode(pi_handle_, pin_b_, PI_INPUT);

      if (gpio_read(pi_handle_, pin_a_))
      {
        state_ |= 1;
      }
      if (gpio_read(pi_handle_, pin_b_))
      {
        state_ |= 2;
      }

      callback_ex(pi_handle_, pin_a_, EITHER_EDGE, &EncoderSensor::updatePosition, this);
      callback_ex(pi_handle_, pin_b_, EITHER_EDGE, &EncoderSensor::updatePosition, this);
    }

    int EncoderSensor::getPosition() const
    {
      return position_;
    }

    void EncoderSensor::updatePosition(int pi_handle, unsigned int gpio, unsigned int level, uint32_t tick, void *userdata)
    {
      EncoderSensor *self = static_cast<EncoderSensor *>(userdata);
      int state = self->state_ & 3;

      if (gpio == self->pin_a_)
      {
        if (level)
        {
          state |= 4;
        }
      }
      else if (gpio == self->pin_b_)
      {
        if (level)
        {
          state |= 8;
        }
      }
      // Shift the state variable to the right by two bits to get the previous state
      self->state_ = state >> 2;
      self->comparePreviousStates(state);
    }

    void EncoderSensor::comparePreviousStates(const int state)
    {
      if (state == 1 || state == 7 || state == 8 || state == 14)
      {
        position_ += 1;
      }
      else if (state == 2 || state == 4 || state == 11 || state == 13)
      {
        position_ -= 1;
      }
      else if (state == 3 || state == 12 || state == 6)
      {
        position_ += 2;
      }
      else if (state == 9)
      {
        position_ -= 2;
      }
    }
  }
}