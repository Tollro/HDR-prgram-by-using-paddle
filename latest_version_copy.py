import os
import cv2
import time
import subprocess
import argparse
import re
import periphery
from periphery import Serial
from periphery import GPIO

# 全局变量控制旋转方向
current_degree = 90  # 初始角度为90°
direction = 1        # 1: 正向旋转（增加角度），-1: 反向旋转（减少角度）
flag = 0
same_times = 0
index = 0
numbers = [None for _ in range(100)]

# 定义GPIO编号
gpio_number = 134  # 根据之前的计算结果

# 导出GPIO
os.system(f"echo {gpio_number} > /sys/class/gpio/export")

# 设置GPIO方向为输入
os.system(f"echo in > /sys/class/gpio/gpio{gpio_number}/direction")

# 尝试设置上拉电阻（如果内核支持）
try:
    os.system(f"echo pullup > /sys/class/gpio/gpio{gpio_number}/bias")
except Exception as e:
    print("设置上拉电阻失败:", str(e))

# 初始化GPIO
button = GPIO(gpio_number, "in")
print("GPIO Value:", button.read())

key = 0    #标志按键

def mg90s_control():
    global current_degree, direction
    # 计算下一步角度
    new_degree = current_degree + direction * 10
    
    # 边界检查并反转方向
    if new_degree >= 180:
        new_degree = 180
        direction = -1  # 到达135°后反向旋转
    elif new_degree <= 0:
        new_degree = 0
        direction = 1   # 到达45°后正向旋转
    else:
        # 未到边界时保持方向
        pass
    
    # 设置舵机角度
    set_servo_degree(new_degree)
    current_degree = new_degree
    time.sleep(0.15)

def set_servo_degree(degree2):
    if degree2 > 180 or degree2 <0:
        duty = 0
        print("degree erro!")
    else:
        time = 0.5 + degree2 * 2.0 / 180.0
        duty = time / 20.0
    pwm.duty_cycle = 1 - duty
    print(f"pwm: {duty}")

def extract_recognize_result(text):
    # 使用正则表达式匹配“regconize result: ”后的内容，直到“score=”之前
    results = []
    lines = text.split("\n")
    for line in lines:
        # 查找“regconize result: ”后跟随的内容，直到“score=”之前
        match = re.search(r'regconize result:\s*([^\s,]+)', line)
        if match:
            results.append(match.group(1))
    return results

def run_ppocr(image_path):
    global flag
    # 设置默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(script_dir, "rknn_ppocr_system_demo")
    det_model = os.path.join(script_dir, "ppocrv4_det_rk3576.rknn")
    rec_model = os.path.join(script_dir, "ppocrv4_rec_rk3576.rknn")
    
    # 检查文件是否存在
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"可执行文件 {demo_path} 不存在")
    if not os.path.exists(det_model):
        raise FileNotFoundError(f"检测模型文件 {det_model} 不存在")
    if not os.path.exists(rec_model):
        raise FileNotFoundError(f"识别模型文件 {rec_model} 不存在")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件 {image_path} 不存在")
    
    # 构建命令
    cmd = [demo_path, det_model, rec_model, image_path]
    
    # 执行命令
    try:
        # print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("命令执行成功!")
        print("输出结果:\n")
        # print(result.stdout)
        times = 0 # 数字出现次数
        chinese_results = extract_recognize_result(result.stdout)
        for chinese_result in chinese_results:
            if chinese_result >= '0'  and chinese_result <= '9':
                times += 1
                digit = chinese_result
            print(chinese_result)
        if times > 1:
            print("检测到多个数字")
        elif times == 1:
            flag = 1
            return digit
    except subprocess.CalledProcessError as e:
        print("命令执行失败!")
        print(f"错误信息: {e.stderr}")

if __name__ == "__main__":

    os.chdir("/home/cat/python_docs/dig_rec/")

    # 初始化串口
    uart = Serial("/dev/ttyS3", baudrate=9600, databits=8, parity="none", stopbits=1)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
    save_quality = 100
    filename = "get.jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, save_quality]
    rate = 1.5
    y_center = int(700*rate/2)
    x_center = int(1200*rate/2)

    pwm = periphery.PWM(chip=0, channel=0)
    frequency = 50; #50 Hz
    pwm.frequency = frequency
    duty_cycle = 0
    pwm.duty_cycle = 1 - duty_cycle
    print(duty_cycle)
    pwm.enable()
    degree = 0
    last_degree = 0
    set_servo_degree(degree)
    time.sleep(0.5)
    stop = 0
    print(f"初始GPIO状态: {button.read()}")
    
    if not cap.isOpened():
        print("无法打开摄像头，请检查驱动和设备连接")
        exit()
    while True:
        
        if not button.read():  # 检测按键按下（低电平）
            print("按键按下")
            time.sleep(0.05)  # 防抖
            while not button.read(): pass  # 等待释放
            key = 1

        while key == 1:
            
            if stop:
                break
            # 丢弃旧帧
            for _ in range(4):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                print("无法获取视频帧，退出...")
                break
            # cv2.imshow('Camera Feed', frame)
            #保存图片
            width = int(frame.shape[1]*rate)
            height = int(frame.shape[0]*rate)
            resized_img = cv2.resize(frame,(width,height),interpolation=cv2.INTER_CUBIC)[y_center-250:y_center+250,x_center-250:x_center+250]
            cv2.imwrite(filename, resized_img, params)

            ##########识别#############
            
            number = run_ppocr(os.path.abspath(filename))
            
            ##########显示检测框##########
            out_img = cv2.imread('./out.jpg')
            # cv2.imshow('Camera Feed', out_img)
            if not flag:
                mg90s_control()
            else:
                print("检测到一个数字，有效")
                numbers[index] = number
                flag = 0
                same_times += 1
                index += 1
                data = "none"

                if same_times >= 2 and numbers[index-1] == numbers[index-2]:
                    print("两次检测数据相同")
                    # 数据通信循环
                    send_times = 0
                    while send_times <= 6: 
                        send_times += 1
                        # 发送数据到目标设备
                        message = str(numbers[index-1])
                        if message:
                            uart.write(message.encode('utf-8'))
                            # uart.write(message)
                            # print(f"已发送数据： {message}")
                            print(f"已发送数据： {message.encode('utf-8', errors='ignore')}")
                            # 简单延时避免过快循环
                            time.sleep(0.2)

                            # 接收来自目标设备的数据
                            data = uart.read(1024, timeout=1)
                            if data:
                                # print(f"接收到的数据: {data}")
                                print(f"接收到的数据: {data.decode('utf-8', errors='ignore')}")
                                data = data.decode('utf-8', errors='ignore')
                                if data == message:
                                    message = "o"
                                    uart.write(message.encode('utf-8'))
                                    print("已发送o！")

                                    time.sleep(0.2)
                                    
                                    data = uart.read(1024, timeout=2)
                                    data = data.decode('utf-8', errors='ignore')
                                    if data == "o":
                                        print("校验成功！")
                                        stop = 1
                                        key = 0
                                        break
                                    else:
                                        print("ok校验失败！")
                                        if send_times >= 5:
                                            print("通讯失败！")
                                            break
                                        continue
                                else:
                                    print("数字校验错误！")
                                    if send_times >= 5:
                                        print("通讯失败！")
                                        break
                                    continue
                            else:
                                print("未收到校验信息！")
                        if data == "o":
                            break
                    mg90s_control()
                                
                else:
                    if index >= 2:
                        print("两次检测数字不同！")
                        index = 0
                        same_times = 0
                    mg90s_control()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
                break

    cap.release()
    cv2.destroyAllWindows()