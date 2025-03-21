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


img_path = './test2_1.png'
# 读取原始图像并显示
im = Image.open(img_path)
plt.imshow(im)
plt.show()
# 将原始图像转为灰度图
im = im.convert('L')
print('原始图像shape: ', np.array(im).shape)
# 使用Image.LANCZOS方式采样原始图片
im = im.resize((28, 28), Image.Resampling.LANCZOS)
plt.imshow(im)
plt.show()
print("采样后图片shape: ", np.array(im).shape)

# 定义预测过程
model = paddle.vision.models.LeNet()
params_file_path = './mnist_model.pdparams'
# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = load_image(img_path)
result = model(paddle.to_tensor(tensor_img))
print('result',result)
#  预测输出取整，即为预测的数字，打印结果
print("本次预测的数字是", np.argmax(result.numpy(), axis=1))
