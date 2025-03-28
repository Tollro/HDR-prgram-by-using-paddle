import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# 初始化 PaddleOCR（启用文本检测+识别）
ocr = PaddleOCR(
    use_angle_cls=True,  # 启用方向分类
    lang='en',           # 英文模型（支持数字识别）
    det_db_thresh=0.3,   # 调低检测阈值以捕捉小文本
    det_db_box_thresh=0.4,
    rec_image_shape='3, 32, 320',  # 优化识别输入尺寸
    #det_model_dir='./det_model_final.pdparams',  # 指定微调后的检测模型参数目录
    #rec_model_dir='./rec_model_final.pdparams',  # 指定微调后的识别模型参数目录
)

def process_digits(img):
    global digit_roi

    # 执行 OCR（检测+识别）
    result = ocr.ocr(img, cls=True)

    # 提取数字区域和识别结果
    digits = []
    for line in result:
        if line is None:
            continue
        for word_info in line:
            text = word_info[1][0]
            if text.isdigit():  # 仅处理数字
                # 获取检测框坐标
                box = np.array(word_info[0]).astype(np.int32)
                
                # 裁剪数字区域
                x_min, y_min = np.min(box, axis=0)
                x_max, y_max = np.max(box, axis=0)
                digit_roi = img[y_min:y_max, x_min:x_max]
                
                # 优化预处理（更适合小数字）
                processed = preprocess_roi(digit_roi)
                
                digits.append({
                    'box': box,
                    'image': processed,
                    'text': text
                })

    # 使用OpenCV可视化结果
    show_boxes = img.copy()
    for d in digits:
        # 在BGR图像上绘制绿色框
        cv2.polylines(show_boxes, [d['box']], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 转换颜色空间用于显示
    original_display = img
    detected_display = show_boxes
    
    # 显示
    cv2.imshow('Digit Detection Results', show_boxes)

    return digits

def preprocess_roi(roi):
    """针对小数字的优化预处理"""
    # 转为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 自适应二值化（避免光照影响）
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 去除小噪点
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 填充边缘留白
    h, w = cleaned.shape
    border = cv2.copyMakeBorder(cleaned, 10,10,10,10, cv2.BORDER_CONSTANT, value=0)
    
    # 保持比例调整大小（适合OCR输入）
    resized = cv2.resize(border, (32, 32), interpolation=cv2.INTER_AREA)
    
    return resized

#########################################主程序###########################################
if __name__ == "__main__":

    # 使用 DirectShow 后端（cv2.CAP_DSHOW）打开摄像头，设备索引一般为 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("无法打开摄像头，请检查驱动和设备连接")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧，退出...")
            break

        cv2.imshow('Camera Feed', frame)
        detected_digits = process_digits(frame)
        # 打印识别结果
        for idx, d in enumerate(detected_digits, 1):
            if(d):
                print("识别到的数字：")
                print(f"数字 {idx}: 坐标={d['box'].tolist()}, OCR识别结果={d['text']}")
        time.sleep(0.05)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break
    
        
    