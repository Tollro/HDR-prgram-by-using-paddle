import os
import cv2
import time
import subprocess
import argparse
import re


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
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("命令执行成功!")
        print("输出结果:")
        # print(result.stdout)
        chinese_results = extract_recognize_result(result.stdout)
        for chinese_result in chinese_results:
            #if chinese_result >=  & chinese_result <= :
                #flag
            #else:
                #flag
            print(chinese_result)
    except subprocess.CalledProcessError as e:
        print("命令执行失败!")
        print(f"错误信息: {e.stderr}")

if __name__ == "__main__":
    
    os.chdir("/home/cat/python_docs/dig_rec/")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    save_quality = 80
    filename = "get.jpg"

    if not cap.isOpened():
        print("无法打开摄像头，请检查驱动和设备连接")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧，退出...")
            break

        # cv2.imshow('Camera Feed', frame)
        #保存图片
        params = [cv2.IMWRITE_JPEG_QUALITY, save_quality]
        cv2.imwrite(filename, frame, params)
        print(f"已保存图片: {filename}")

        ##########识别#############
        run_ppocr(os.path.abspath(filename))
        ##########显示检测框##########
        out_img = cv2.imread('./out.jpg')
        cv2.imshow('Camera Feed', out_img)




        time.sleep(0.1)


        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()