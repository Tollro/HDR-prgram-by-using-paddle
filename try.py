import paddlehub as hub
import cv2

# 加载 PaddleHub 提供的 MNIST 模块（预训练模型）
mnist = hub.Module(name="mnist")

# 读取待识别的数字图像（建议为灰度图），这里假设图像路径为 'digit.png'
img = cv2.imread('./test4_1.png', cv2.IMREAD_GRAYSCALE)

# MNIST 模型要求图像尺寸为 28x28，如果图像尺寸不符合则需调整
img = cv2.resize(img, (28, 28))

# 进行数字识别，输入需要以列表形式传入（支持批量预测）
results = mnist.recognize([img])

# 输出识别结果
print("识别结果:", results)