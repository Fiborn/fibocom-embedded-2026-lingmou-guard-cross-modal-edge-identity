#include "servo.h"
#include "stm32f10x_tim.h"  
#include "stm32f10x_gpio.h"
#include "stm32f10x_rcc.h"
#include "misc.h"

void TIM4_PWM_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
 	TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	TIM_OCInitTypeDef  TIM_OCInitStructure;
	
	// 1. 开启时钟（TIM4+GPIOB）
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM4, ENABLE);	// TIM4挂在APB1，时钟72M
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	
	// 2. 配置GPIO为复用推挽输出（PB6=TIM4_CH1、PB7=TIM4_CH2）
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_7;  // 同时配置PB6和PB7
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;         // 复用推挽输出
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOB, &GPIO_InitStructure);
	 
	// 3. 配置定时器时基（50Hz PWM：72M/(7200-1)/(200-1) = 50Hz）
	TIM_TimeBaseStructure.TIM_Period = 199;				 // 自动重装载值：200-1=199（周期20ms）
	TIM_TimeBaseStructure.TIM_Prescaler = 7199;			 // 预分频器：7200-1=7199（72M/7200=10KHz）
	TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);
	
	// 4. 配置通道1（PB6）PWM输出
	TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;		 // PWM模式1：CNT < CCR时输出高电平
	TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
	TIM_OCInitStructure.TIM_Pulse = 10;						 // 初始占空比（1ms高电平，0°）
	TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;	 // 高电平有效
	TIM_OC1Init(TIM4, &TIM_OCInitStructure);			   	 
	TIM_OC1PreloadConfig(TIM4, TIM_OCPreload_Enable);		 // 使能通道1预装载
	
	// 5. 配置通道2（PB7）PWM输出
	TIM_OCInitStructure.TIM_Pulse = 10;						 // 初始占空比（1ms高电平，0°）
	TIM_OC2Init(TIM4, &TIM_OCInitStructure);			   	 
	TIM_OC2PreloadConfig(TIM4, TIM_OCPreload_Enable);		 // 使能通道2预装载
	
	// 6. 使能定时器
	TIM_Cmd(TIM4, ENABLE);					
}

/*以下函数为0-180度舵机角度控制测试函数
	*	PWM 信号与0-180舵机的关系：
	*	0.5ms ---------------- 0度
	*	1ms   ---------------- 45度
	*	1.5ms ---------------- 90度
	*	2ms   ---------------- 135度
	*	2.5ms ---------------- 180度
 
	*	舵机频率与占空比的计算：
	*	设舵机的频率为50HZ，则PWM周期为20ms，0度对应的占空比为2.5%，即0.05ms的高电平输出。
 */
 
// PB6
void Servo1_Set_Angle_Float(float angle)
{
	u16 ccr;  
	if(angle < 0.00f)
	{
		angle = 0.00f;
	}
	else if(angle > 180.00f)
	{
		angle = 180.00f;
	}
	/*
	5  0度
	15 90度
	25 180度
	*/
	ccr = 5 + (u16)(angle * 20.0f / 180.0f+0.5f);
//	ccr = 10 + (u16)(angle * 10.0f / 90.0f + 0.5f);
	
	// 设置CCR值（TIM4_CH1）
	TIM_SetCompare1(TIM4, ccr);
}

// PB7
void Servo2_Set_Angle_Float(float angle)
{
	u16 ccr; 
	if(angle < 0.00f)
	{
		angle = 0.00f;
	}
	else if(angle > 180.00f)  // 
	{
		angle = 180.00f;
	}
	ccr = 5 + (u16)(angle * 20.0f / 180.0f+0.5f);  
	TIM_SetCompare2(TIM4, ccr);
}