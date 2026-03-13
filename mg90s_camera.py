import os
import cv2
import time
import subprocess
import argparse
import re
import periphery

# 全局变量控制旋转方向
current_degree = 90  # 初始角度为90°
direction = 1        # 1: 正向旋转（增加角度），-1: 反向旋转（减少角度）
flag = 0
flag_times = 0

def mg90s_control():
    global current_degree, direction
    # 计算下一步角度
    new_degree = current_degree + direction * 15
    
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

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
    save_quality = 100
    filename = "get.jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, save_quality]
    rate = 1.5
    y_center = int(700*rate//2)
    x_center = int(1200*rate//2)
    
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
    i = 0

    
    if not cap.isOpened():
        print("无法打开摄像头，请检查驱动和设备连接")
        exit()

    while True:
        # 丢弃旧帧
        for _ in range(4):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧，退出...")
            break
        cv2.imshow('Camera Feed', frame)
        #保存图片
        width = int(frame.shape[1]*rate)
        height = int(frame.shape[0]*rate)
        resized_img = cv2.resize(frame,(width,height),interpolation=cv2.INTER_CUBIC)[y_center-250:y_center+250,x_center-250:x_center+250]
        cv2.imwrite(filename, resized_img, params)
        ##########识别#############
        # time.sleep(1)
        number = run_ppocr(os.path.abspath(filename))
        # time.sleep(3)
        ##########显示检测框##########
        out_img = cv2.imread('./out.jpg')
        cv2.imshow('OUT PUT', out_img)
        if not flag:
            mg90s_control()
        else:

            print("检测到一个数字，有效")

            flag = 0
            flag_times += 1
            if flag_times >= 2:
                break
        
        
        # print(f"已保存图片: {filename}")

        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()