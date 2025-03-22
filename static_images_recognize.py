import cv2
import paddle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # print(np.array(im))
    im = im.resize((28, 28), Image.Resampling.LANCZOS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 将图像转换为 numpy 数组，并转换数据类型为 float32
    im = np.array(im).astype(np.float32)
    # 归一化像素值到 [0, 1]，并反转颜色（1 - im/255）
    im = 1 - im / 255.
    # 添加批量和通道两个维度，使得最终形状为 [1, 1, 28, 28]
    im = im.reshape(1, 1, 28, 28)
    return im

def enhanced_preprocess(img_path, sharpen_strength=1.5, contrast_alpha=1.5, contrast_beta=30):
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
    gaussian_3 = cv2.GaussianBlur(blurred, (0, 0), 2.0)
    sharpened = cv2.addWeighted(blurred, sharpen_strength, gaussian_3, -0.5, 0)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(sharpened, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作（去除小噪点）
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 调整尺寸
    resized = cv2.resize(processed, (28, 28))
    
    cv2.imshow('处理后图像',img)
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
    
    cv2.imshow('Original', img)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化反转（MNIST风格：白字黑底）
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 调整尺寸为28x28
    resized = cv2.resize(thresh, (28, 28))
    
    cv2.imshow('Processed', img)

    # 归一化并转换数据类型
    normalized = resized.astype('float32') / 255.0
    
    # 调整维度为 [C, H, W] 并添加batch维度
    input_tensor = normalized[np.newaxis, np.newaxis, :, :]
    
    return input_tensor

def predict_single_image(img_path):
    # 预处理图像
    input_tensor = enhanced_preprocess(img_path)
    #input_tensor = preprocess_image(img_path)
    
    # 转换为Paddle Tensor
    input_data = paddle.to_tensor(input_tensor)
    
    # 预测
    with paddle.no_grad():
        output = model(input_data)
        print('result',output)
        prediction = paddle.argmax(output, axis=1).numpy()[0]
    
    return prediction


#####################主程序#####################
img_path = './test1_1.png'
model = paddle.vision.models.LeNet()
model_state_dict = paddle.load('./mnist_model.pdparams')  # 替换为你的模型路径
model.set_state_dict(model_state_dict)
model.eval()

pred = predict_single_image(img_path)
print(f"Predicted digit: {pred}")

# # 读取原始图像并显示
# im = cv2.imread(img_path)
# cv2.imshow('处理前图像',im)
# #tensor_img = load_image(img_path) # 此处换用不同的函数实现
# processed_img = process_image_full(img_path)
# # 扩展为 4-D 张量：[batch_size, channels, height, width]
# tensor_img_4d = processed_img.reshape(1, 1, processed_img.shape[0], processed_img.shape[1])
# print("新的形状：", tensor_img_4d.shape)
# # 定义预测过程
# model = paddle.vision.models.LeNet()
# params_file_path = './mnist_model.pdparams'
# # 加载模型参数
# param_dict = paddle.load(params_file_path)
# model.load_dict(param_dict)
# # 灌入数据
# model.eval()
# result = model(paddle.to_tensor(tensor_img_4d))
# print('result',result)
# #  预测输出取整，即为预测的数字，打印结果
# print("本次预测的数字是", np.argmax(result.numpy(), axis=1))
