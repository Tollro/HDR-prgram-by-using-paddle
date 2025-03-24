import paddle
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

# # 定义 LeNet 模型
# class LeNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5)
#         self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
#         self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
#         self.fc1 = paddle.nn.Linear(in_features=16 * 4 * 4, out_features=120)
#         self.fc2 = paddle.nn.Linear(in_features=120, out_features=84)
#         self.fc3 = paddle.nn.Linear(in_features=84, out_features=10)
    
#     def forward(self, x):
#         x = self.pool(paddle.nn.functional.relu(self.conv1(x)))
#         x = self.pool(paddle.nn.functional.relu(self.conv2(x)))
#         x = paddle.flatten(x, 1)
#         x = paddle.nn.functional.relu(self.fc1(x))
#         x = paddle.nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# 初始化模型
model = paddle.vision.models.LeNet()

# 加载已保存的模型参数文件（例如 'lenet_model.pdparams'）
model_state_dict = paddle.load('./mnist_model.pdparams')
model.set_state_dict(model_state_dict)
model.eval()  # 设置为评估模式

# # 加载 MNIST 测试数据集
# test_dataset = MNIST(mode='test', transform=ToTensor())
# test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载 MNIST 测试数据集
transform = paddle.vision.transforms.Compose([
    paddle.vision.transforms.ToTensor(),
    paddle.vision.transforms.Normalize(mean=[0.5], std=[0.5])
])
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

# 随机选择一些样本进行可视化验证
num_samples = 5  # 设置要可视化的样本数量
indices = np.random.choice(len(test_dataset), num_samples, replace=False)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(indices):
    img, label = test_dataset[idx]
    img_input = img.unsqueeze(0)  # 增加 batch 维度
    output = model(img_input)
    pred_label = output.argmax(axis=1).numpy()[0]

    # 可视化
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img.squeeze().numpy(), cmap='gray')
    plt.title(f'Pred: {pred_label}\nTrue: {label}')
    plt.axis('off')

plt.show()

# # 验证过程：统计预测正确的数量
# correct = 0
# total = 0
# for images, labels in test_loader:
#     outputs = model(images)
#     pred = outputs.argmax(axis=1)
#     labels = paddle.squeeze(labels, axis=1) if len(labels.shape) > 1 else labels
#     correct += (pred == labels).numpy().sum()
#     total += labels.shape[0]

# accuracy = correct / total * 100
# print("Test Accuracy: {:.2f}%".format(accuracy))