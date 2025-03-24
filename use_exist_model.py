import cv2
import paddle
import numpy as np
import paddlehub as hub


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


# 预处理函数（需调整输入为 224x224 RGB）
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 转 RGB
    img = cv2.resize(img, (224, 224))
    img = (img.astype('float32') - 127.5) / 127.5  # 归一化到 [-1,1]
    return img.transpose([2, 0, 1])[np.newaxis, :]  # 调整维度为 [1,3,224,224]


#####################主程序#####################
#使用官方提供的参数
img_path = './test4_1.png'
input_tensor = preprocess_image(img_path)

model = hub.Module(name="resnet50_vd_ssld",inputsize=224) #直接调用内置resnet
model.fc = hub.LinearLayer(input_dim=2048, output_dim=10) #修改输出层为10分类
model.load_parameters('resnet50_mnist.pdparams')          #加载mnist与训练参数

result = model(input_tensor)
prediction = np.argmax(result.numpy())
print(f"预测结果: {prediction}")


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
