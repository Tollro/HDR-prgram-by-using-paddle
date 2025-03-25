import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. 数据预处理与加载
# -------------------------------
from paddle.vision.transforms import Compose, ToTensor, Normalize
# 对 MNIST 图片（灰度图）进行归一化处理，输出形状 [1, 28, 28]
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# 加载 MNIST 数据集，作为识别任务数据（每张图片对应一个数字标签）
train_dataset_rec = paddle.vision.datasets.MNIST(mode="train", transform=transform)
test_dataset_rec = paddle.vision.datasets.MNIST(mode="test", transform=transform)

# -------------------------------
# 2. 构建 CRNN 模型用于识别任务
# -------------------------------
class CRNN(nn.Layer):
    def __init__(self, num_classes):
        """
        :param num_classes: 类别数，CTC需要包括空白符，若识别数字0-9，则num_classes=10+1=11
        """
        super(CRNN, self).__init__()
        # CNN 部分：提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv2D(1, 64, kernel_size=3, padding=1),   # 输入[1,28,28] -> 输出[64,28,28]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),          # [64,14,14]
            nn.Conv2D(64, 128, kernel_size=3, padding=1),    # [128,14,14]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)           # [128,7,7]
        )
        # 将 CNN 输出 reshape 成序列形式：
        # 将 [batch, 128,7,7] reshape 成 [batch, time_steps, features]，
        # 这里将宽度（7）作为时间步，每步的特征维度为 128*7
        # RNN 部分：双向LSTM
        self.lstm = nn.LSTM(input_size=128*7, hidden_size=256, num_layers=2, direction='bidirectional')
        # 全连接层将 RNN 输出映射到类别数
        self.fc = nn.Linear(in_features=256*2, out_features=num_classes)
    
    def forward(self, x):
        """
        :param x: 输入图片，形状 [batch, 1, 28, 28]
        :return: logits，形状 [time_steps, batch, num_classes]（CTC要求）
        """
        conv = self.cnn(x)  # 输出形状 [batch, 128, 7, 7]
        batch_size, channels, height, width = conv.shape
        # reshape：将通道和高度合并，宽度作为时间步
        conv = conv.reshape([batch_size, channels * height, width])  # [batch, 128*7, 7]
        conv = conv.transpose([0, 2, 1])  # [batch, 7, 128*7]
        # LSTM 处理序列，输出 shape: [batch, time, 512]
        lstm_out, _ = self.lstm(conv)
        logits = self.fc(lstm_out)  # [batch, time, num_classes]
        # 转换维度为 [time, batch, num_classes] 供 CTC 损失使用
        logits = logits.transpose([1, 0, 2])
        return logits

# 定义类别数：数字 0-9 + 空白符 = 11
num_classes = 11
rec_model = CRNN(num_classes=num_classes)

# 如果有微调后预训练权重，可以加载：
# pretrained_dict = paddle.load("path/to/fine_tuned_rec_model.pdparams")
# rec_model.set_state_dict(pretrained_dict)

# -------------------------------
# 3. 定义 CTC 损失和优化器
# -------------------------------
ctc_loss = nn.CTCLoss(blank=0)  # 通常将索引0作为空白符
optimizer_rec = optim.Adam(parameters=rec_model.parameters(), learning_rate=0.001)

# -------------------------------
# 4. 创建数据加载器
# -------------------------------
train_loader_rec = DataLoader(train_dataset_rec, batch_size=64, shuffle=True)
test_loader_rec = DataLoader(test_dataset_rec, batch_size=64, shuffle=False)

# -------------------------------
# 5. 训练函数
# -------------------------------
# 在训练函数和评估函数中，将logit_length和label_length的构造调整为正确的形状和数据类型
def train_rec_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch_id, (img, label) in enumerate(loader):
        label = label.flatten().astype('int64')  # 确保标签是int64
        logits = model(img)
        # 确保logit_length是一维，且dtype=int64
        logit_length = paddle.full([logits.shape[1]], logits.shape[0], dtype='int64')
        # 确保label_length是一维，且dtype=int64
        label_length = paddle.full([logits.shape[1]], 1, dtype='int64')
        loss = loss_fn(logits, label, logit_length, label_length)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss += loss.item()
        if batch_id % 100 == 0:
            print(f"[Recognition] Batch {batch_id}: Loss = {loss.item():.6f}")
    return total_loss / len(loader)

def evaluate_rec(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with paddle.no_grad():
        for img, label in loader:
            label = label.flatten().astype('int64')
            logits = model(img)
            logit_length = paddle.full([logits.shape[1]] * logits.shape[0], dtype='int64')
            label_length = paddle.full([logits.shape[1]], 1, dtype='int64')
            loss = loss_fn(logits, label, logit_length, label_length)
            total_loss += loss.item()
            # 简单预测：取第一个时间步的最大概率作为预测
            pred = logits[0].argmax(axis=1)
            total_correct += (pred == label).sum().item()
            total_samples += img.shape[0]
    return total_loss / len(loader), total_correct / total_samples

# -------------------------------
# 6. 训练识别模型
# -------------------------------
num_epochs_rec = 5
for epoch in range(num_epochs_rec):
    avg_loss = train_rec_epoch(rec_model, train_loader_rec, optimizer_rec, ctc_loss)
    val_loss, val_acc = evaluate_rec(rec_model, test_loader_rec, ctc_loss)
    print(f"Epoch {epoch+1}/{num_epochs_rec}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}, Val Acc = {val_acc:.4f}")
# 对识别模型保存训练参数
paddle.save(rec_model.state_dict(), "rec_model_final.pdparams")

# -------------------------------
# 7. 推理测试与可视化
# -------------------------------
rec_model.eval()
img, true_label = test_dataset_rec[0]
img_batch = img.unsqueeze(0)  # 增加 batch 维度 [1, 1, 28, 28]
logits = rec_model(img_batch)   # 输出 shape: [time, 1, num_classes]
# 简单取第一个时间步的预测结果
pred = logits[0].argmax(axis=1).numpy()[0]
print(f"True Label: {true_label[0]}, Predicted Label: {pred}")

# 使用 matplotlib 可视化
plt.imshow(img[0], cmap='gray')
plt.title(f"True: {true_label[0]}, Pred: {pred}")
plt.show()
