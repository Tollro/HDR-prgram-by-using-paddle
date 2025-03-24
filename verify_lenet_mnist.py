import paddle
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor

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

# 加载 MNIST 测试数据集
test_dataset = MNIST(mode='test', transform=ToTensor())
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 验证过程：统计预测正确的数量
correct = 0
total = 0
for images, labels in test_loader:
    outputs = model(images)
    pred = outputs.argmax(axis=1)
    labels = paddle.squeeze(labels, axis=1) if len(labels.shape) > 1 else labels
    correct += (pred == labels).numpy().sum()
    total += labels.shape[0]

accuracy = correct / total * 100
print("Test Accuracy: {:.2f}%".format(accuracy))