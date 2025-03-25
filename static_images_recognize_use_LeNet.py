import cv2
import paddle
import matplotlib.pyplot as plt
import numpy as np


def enhanced_preprocess(img_path, sharpen_strength=0.5, contrast_alpha=2, contrast_beta=5):
    """
    改进的预处理流程：
    1. 读取图像
    2. 转灰度
    3. 对比度增强
    4. 锐化处理
    5. 自适应阈值
    6. 形态学操作
    7. 尺寸调整
    8. 归一化
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or invalid path")
    
    cv2.imshow('处理前图像',img)
    # 等待用户按键（0 表示无限等待）
    cv2.waitKey(0)
    # 关闭所有 OpenCV 创建的窗口
    cv2.destroyAllWindows()

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 对比度增强 (线性变换)
    contrasted = cv2.convertScaleAbs(gray, alpha=contrast_alpha, beta=contrast_beta)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(contrasted, (5, 5), 0)
    
    # 锐化处理 - 使用非锐化掩模(Unsharp Mask)
    gaussian_3 = cv2.GaussianBlur(blurred, (0, 0), 1.8)
    sharpened = cv2.addWeighted(blurred, sharpen_strength, gaussian_3, -0.5, 0)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(sharpened, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 1.3)
    
    # 形态学操作（去除小噪点）
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=1)
    
    # 调整尺寸
    resized = cv2.resize(processed, (28, 28))
    
    cv2.imshow('处理后图像',processed)
    # 等待用户按键（0 表示无限等待）
    cv2.waitKey(0)
    # 关闭所有 OpenCV 创建的窗口
    cv2.destroyAllWindows()

    # 归一化处理
    normalized = resized.astype('float32') / 255.0
    
    # 调整维度 [C, H, W] + batch维度
    return normalized[np.newaxis, np.newaxis, :, :]


#普通处理图像
def preprocess_image(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or invalid path")
    
    cv2.imshow('处理前图像',img)
    # 等待用户按键（0 表示无限等待）
    cv2.waitKey(0)
    # 关闭所有 OpenCV 创建的窗口
    cv2.destroyAllWindows()
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化反转（MNIST风格：白字黑底）
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 调整尺寸为28x28
    resized = cv2.resize(thresh, (28, 28))
    
    cv2.imshow('处理后图像',thresh)
    # 等待用户按键（0 表示无限等待）
    cv2.waitKey(0)
    # 关闭所有 OpenCV 创建的窗口
    cv2.destroyAllWindows()

    # 归一化并转换数据类型
    normalized = resized.astype('float32') / 255.0
    
    # 调整维度为 [C, H, W] 并添加batch维度
    input_tensor = normalized[np.newaxis, np.newaxis, :, :]
    
    return input_tensor

def predict_single_image(img_path):
    # 预处理图像
    #input_tensor = enhanced_preprocess(img_path)
    input_tensor = preprocess_image(img_path)
    
    # 转换为Paddle Tensor
    input_data = paddle.to_tensor(input_tensor)
    
    # 预测
    with paddle.no_grad():
        output = model(input_data)
        print('result',output)
        prediction = paddle.argmax(output, axis=1).numpy()[0]
    
    return prediction


#####################主程序#####################
img_path = './new_tests/3_0.jpg'
model = paddle.vision.models.LeNet()
model_state_dict = paddle.load('./mnist_model.pdparams')  # 替换为你的模型路径
model.set_state_dict(model_state_dict)
model.eval()

pred = predict_single_image(img_path)
print(f"Predicted digit: {pred}")
