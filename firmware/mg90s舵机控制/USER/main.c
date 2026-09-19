#include "delay.h"
#include "sys.h"
#include "usart.h"
#include "servo.h"
#include "stdlib.h"
#include "string.h"

#define SERVO_HOLD_DELAY 100   

float target_angle1 = 0.00f;
float target_angle2 = 0.00f;

u8 Parse_Servo_Angle(u8 *buf, u16 len, float *angle1, float *angle2)
{
    u8 temp_buf[32] = {0};
    u16 i = 0, j = 0;
    u8 servo_id = 0;
    float angle = 0.00f;

    memset(temp_buf, 0, sizeof(temp_buf));

    for(i = 0; i < len - 1; i++)
    {
        if(buf[i] == '1' && buf[i + 1] == '_')
        {
            servo_id = 1;
            i += 2;
            break;
        }
        else if(buf[i] == '2' && buf[i + 1] == '_')
        {
            servo_id = 2;
            i += 2;
            break;
        }
    }

    for(; i < len; i++)
    {
        if((buf[i] >= '0' && buf[i] <= '9') || buf[i] == '.')
        {
            temp_buf[j++] = buf[i];
            if(j >= 31) break;
        }
    }

    angle = atof((char*)temp_buf);

    if(servo_id == 1)
    {
        if(angle < 0) angle = 0;
        if(angle > 180) angle = 180;
        *angle1 = angle;
        return 1;
    }
    else if(servo_id == 2)
    {
        if(angle < 0) angle = 0;
        if(angle > 180) angle = 180;
        *angle2 = angle;
        return 2;
    }

    return 0;
}

void Servo_Move_And_Off(u8 servo_id, float angle, u16 hold_ms)
{
    if(servo_id == 1)
    {
        // 1. 开启 PWM 输出
//        TIM_CCxCmd(TIM4, TIM_Channel_1, TIM_CCx_Enable);
//        Servo1_Set_Angle_Float(angle);
//        // 2. 保持指定时间，等待舵机转到指定位置
//        delay_ms(hold_ms);
//        // 3. 卸力：关闭 PWM 输出，防止舵机持续抖动
//        TIM_CCxCmd(TIM4, TIM_Channel_1, TIM_CCx_Disable);
			 Servo1_Set_Angle_Float(angle);
    }
    else if(servo_id == 2)
    {
//        TIM_CCxCmd(TIM4, TIM_Channel_2, TIM_CCx_Enable);
//        Servo2_Set_Angle_Float(angle);
//        delay_ms(hold_ms);
//        TIM_CCxCmd(TIM4, TIM_Channel_2, TIM_CCx_Disable);
			 Servo2_Set_Angle_Float(angle);
    }
}
// servo1:Y轴 PB6
// servo2:X轴 PB7
int main(void)
{
    u16 len;
    static float current_angle1 = 0.0f;
    static float current_angle2 = 0.0f;
    u8 res;

    delay_init();
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    uart_init(115200);
    TIM4_PWM_Init();
		TIM_CCxCmd(TIM4, TIM_Channel_1, TIM_CCx_Enable);
    TIM_CCxCmd(TIM4, TIM_Channel_2, TIM_CCx_Enable);
//		TIM_CCxCmd(TIM4, TIM_Channel_1, TIM_CCx_Disable);
//		TIM_CCxCmd(TIM4, TIM_Channel_2, TIM_CCx_Disable);
		delay_ms(SERVO_HOLD_DELAY);
	  Servo1_Set_Angle_Float(80);
    Servo2_Set_Angle_Float(100);
    delay_ms(SERVO_HOLD_DELAY);
    printf("STM32Init\r\n");

    while(1)
    {
        if(USART_RX_STA & 0x8000)
        {
            len = USART_RX_STA & 0x3FFF;

            res = Parse_Servo_Angle(USART_RX_BUF, len, &target_angle1, &target_angle2);

            if(res == 1)
            {
                current_angle1 = target_angle1;
                Servo_Move_And_Off(1, current_angle1, SERVO_HOLD_DELAY);
                printf("X(PB6): %.2f°\r\n", current_angle1);
                printf("OK\r\n");
            }
            else if(res == 2)
            {
                current_angle2 = target_angle2;
                Servo_Move_And_Off(2, current_angle2, SERVO_HOLD_DELAY);
                printf("Y(PB7): %.2f°\r\n", current_angle2);
                printf("OK\r\n");
            }

            USART_RX_STA = 0;
        }

        delay_ms(10);
    }
}

