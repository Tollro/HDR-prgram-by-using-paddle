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

def process_digits(image_path):
    global digit_roi
    # 读取图像
    img = cv2.imread(image_path)
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    detected_display = cv2.cvtColor(show_boxes, cv2.COLOR_RGB2BGR)
    
    # 添加文字标注
    cv2.putText(original_display, 'Original Image', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(detected_display, 'Detected Digits', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 水平拼接对比显示
    combined = cv2.hconcat([original_display, detected_display])
    
    # 显示并等待
    cv2.imshow('Digit Detection Results', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
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
    image_path = "./new_tests/6_0.jpg"  # 替换为实际图片路径
    detected_digits = process_digits(image_path)
    
    # 打印识别结果
    print("识别到的数字：")
    for idx, d in enumerate(detected_digits, 1):
        print(f"数字 {idx}: 坐标={d['box'].tolist()}, OCR识别结果={d['text']}")

#########################################LeNet区域#######################################
# import paddle
# #普通处理图像
# def preprocess_image(img):
    
#     if img is None:
#         raise ValueError("Image not found or invalid path")
    
#     cv2.imshow('处理前图像',img)
#     # 等待用户按键（0 表示无限等待）
#     cv2.waitKey(0)
#     # 关闭所有 OpenCV 创建的窗口
#     cv2.destroyAllWindows()
    
#     # 转换为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # 二值化反转（MNIST风格：白字黑底）
#     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
#     # 调整尺寸为28x28
#     resized = cv2.resize(thresh, (28, 28))
    
#     cv2.imshow('处理后图像',thresh)
#     # 等待用户按键（0 表示无限等待）
#     cv2.waitKey(0)
#     # 关闭所有 OpenCV 创建的窗口
#     cv2.destroyAllWindows()

#     # 归一化并转换数据类型
#     normalized = resized.astype('float32') / 255.0
    
#     # 调整维度为 [C, H, W] 并添加batch维度
#     input_tensor = normalized[np.newaxis, np.newaxis, :, :]
    
#     return input_tensor

# def predict_single_image(img):
#     # 预处理图像
#     input_tensor = preprocess_image(img)
    
#     # 转换为Paddle Tensor
#     input_data = paddle.to_tensor(input_tensor)
    
#     # 预测
#     with paddle.no_grad():
#         output = model(input_data)
#         print('result',output)
#         prediction = paddle.argmax(output, axis=1).numpy()[0]
    
#     return prediction

# model = paddle.vision.models.LeNet()
# model_state_dict = paddle.load('./mnist_model.pdparams')  # 替换为你的模型路径
# model.set_state_dict(model_state_dict)
# model.eval()

# pred = predict_single_image(digit_roi)
# print(f"LeNet识别结果: {pred}")q
