import periphery
import time

# #打开gpio3_A1
# gpio = periphery.GPIO(97,"out")

# while True:
#     #设置高电平
#     gpio.write(True)
#     level = gpio.read()
#     print(f"level={level}")
#     time.sleep(1)

#     gpio.write(False)
#     #读取电平
#     level = gpio.read()
#     print(f"level={level}")
#     time.sleep(1)

#######呼吸灯示例##########
step = 0.05
pwm = periphery.PWM(chip=0, channel=0)

try:
    frequency = 1000; #1 KHz
    pwm.frequency = frequency
    duty_cycle = 0
    pwm.duty_cycle = duty_cycle
    pwm.enable()

    # 响应10秒
    start = time.time()

    #设置占空比（0~1）
    while time.time() - start < 10:  # 运行 10 秒:
        duty_cycle = 0
        for i in range (20):
            duty_cycle += step
            duty_cycle = min(duty_cycle, 1.0)  # 确保不超过 1.0
            pwm.duty_cycle = duty_cycle
            print(duty_cycle)
            time.sleep(0.05)
        for i in range (20):
            duty_cycle -= step
            duty_cycle = max(duty_cycle, 0.0)  # 确保不低于 0.0
            pwm.duty_cycle = duty_cycle
            print(duty_cycle)
            time.sleep(0.05)

finally:
    # 停用 PWM 输出
    pwm.disable()
    print("PWM 已停用")

    # 关闭 PWM
    pwm.close()
    print("PWM 已关闭")
