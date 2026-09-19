#ifndef __SERVO_H
#define __SERVO_H

#include "stm32f10x.h"

void TIM4_PWM_Init(void);          // 初始化TIM4双通道PWM
void Servo1_Set_Angle_Float(float angle);  // PB6，0.00°~90.00°
void Servo2_Set_Angle_Float(float angle);  // PB7，0.00°~90.00°
#endif