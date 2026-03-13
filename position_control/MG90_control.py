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
def set_servo_degree(degree):
    if degree > 180 or degree <0:
        duty = 0
    else:
        time = 0.5 + degree * 2.0 / 180.0
        duty = time / 20.0
    pwm.duty_cycle = 1 - duty
    print(duty)

####### mg90s示例 ##########
#step = 0.05
pwm = periphery.PWM(chip=0, channel=0)

# 周期 20ms
# 0.5ms → 0°（最小角度）
# 1.5ms → 90°（中间位置)
# 2.5ms → 180°（最大角度）

try:
    frequency = 50; #50 Hz
    pwm.frequency = frequency
    duty_cycle = 0
    pwm.duty_cycle = 1 - duty_cycle
    print(duty_cycle)
    pwm.enable()

    set_servo_degree(90)
    time.sleep(0.5)

finally:
    duty_cycle = 0
    pwm.duty_cycle = 1 - duty_cycle
    # 停用 PWM 输出
    pwm.disable()
    print("PWM 已停用")

    # 关闭 PWM
    pwm.close()
    print("PWM 已关闭")


#  # 响应10秒
#  start = time.time()
#     #设置占空比（0~1）
#     while time.time() - start < 10:  # 运行 10 秒:
#         duty_cycle = 0
#         for i in range (20):
#             duty_cycle += step
#             duty_cycle = min(duty_cycle, 1.0)  # 确保不超过 1.0
#             pwm.duty_cycle = duty_cycle
#             print(duty_cycle)
#             time.sleep(0.05)
#         for i in range (20):
#             duty_cycle -= step
#             duty_cycle = max(duty_cycle, 0.0)  # 确保不低于 0.0
#             pwm.duty_cycle = duty_cycle
#             print(duty_cycle)
#             time.sleep(0.05)

# finally:
#     # 停用 PWM 输出
#     pwm.disable()
#     print("PWM 已停用")

#     # 关闭 PWM
#     pwm.close()
#     print("PWM 已关闭")
