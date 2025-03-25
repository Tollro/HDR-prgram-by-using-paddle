import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# 1. 构造自定义数据集，用于检测任务
# -------------------------------
class MNISTDetectionDataset(Dataset):
    """
    使用 MNIST 数据集构造手写数字检测数据集
    每张图片转换为3通道，并生成一个合成的边框标签
    边框标签格式为：[x1,y1, x2,y2, x3,y3, x4,y4]，此处假设数字区域为整张图片
    """
    def __init__(self, mode="train", transform=None):
        # 直接使用 PaddlePaddle 内置的 MNIST 数据集（灰度图）
        self.mnist = paddle.vision.datasets.MNIST(mode=mode, transform=transform)
        self.mode = mode
        
    def __getitem__(self, index):
        # 获取 MNIST 图片和标签（此处标签原本为数字，但我们不需要分类标签）
        img, label = self.mnist[index]
        # MNIST图片shape为 [1, H, W]，复制通道转换为3通道
        img_3ch = paddle.concat([img, img, img], axis=0)  # shape: [3, 28, 28]
        # 合成边框标签：此处取整张图片区域
        # 注意：这里假设图片尺寸固定为28x28
        bbox = np.array([0, 0, 27, 0, 27, 27, 0, 27], dtype="float32")
        return img_3ch, bbox

    def __len__(self):
        return len(self.mnist)

# 定义图像预处理：将图片归一化至[0,1]
from paddle.vision.transforms import Compose, ToTensor, Normalize

transform = Compose([
    ToTensor(),  # 将图像转换为Tensor，原始MNIST图像已为[1,28,28]
    Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
])

# 创建训练和测试数据集（检测任务）
train_dataset_det = MNISTDetectionDataset(mode="train", transform=transform)
test_dataset_det = MNISTDetectionDataset(mode="test", transform=transform)

# -------------------------------
# 2. 构建简单的检测网络
# -------------------------------
class SimpleDetModel(nn.Layer):
    def __init__(self):
        super(SimpleDetModel, self).__init__()
        # 采用简单的CNN回归网络，输入：[3,28,28]，输出8个数字（边框坐标）
        self.features = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, padding=1),  # [32,28,28]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),         # [32,14,14]
            nn.Conv2D(32, 64, kernel_size=3, padding=1),    # [64,14,14]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)          # [64,7,7]
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8个输出对应四个点的(x, y)坐标
        )
    
    def forward(self, x):
        x = self.features(x)
        out = self.regressor(x)
        return out

# 实例化模型
det_model = SimpleDetModel()

# -------------------------------
# 3. 定义损失函数和优化器
# -------------------------------
criterion_det = nn.MSELoss()
optimizer_det = optim.Adam(parameters=det_model.parameters(), learning_rate=0.001)

# -------------------------------
# 4. 创建数据加载器
# -------------------------------
train_loader_det = DataLoader(train_dataset_det, batch_size=64, shuffle=True)
test_loader_det = DataLoader(test_dataset_det, batch_size=64, shuffle=False)

# -------------------------------
# 5. 训练与评估函数
# -------------------------------
def train_det_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch_id, (img, bbox_gt) in enumerate(loader):
        # img: [batch, 3, 28, 28]  bbox_gt: [batch, 8]
        pred_bbox = model(img)  # 预测边框
        loss = loss_fn(pred_bbox, paddle.to_tensor(bbox_gt))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss += loss.item()
        if batch_id % 100 == 0:
            print(f"[Detection] Batch {batch_id}: Loss = {loss.item():.6f}")
    return total_loss / len(loader)

def evaluate_det(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with paddle.no_grad():
        for img, bbox_gt in loader:
            pred_bbox = model(img)
            loss = loss_fn(pred_bbox, paddle.to_tensor(bbox_gt))
            total_loss += loss.item()
    return total_loss / len(loader)

# -------------------------------
# 6. 训练检测模型
# -------------------------------
num_epochs_det = 5
for epoch in range(num_epochs_det):
    avg_loss = train_det_epoch(det_model, train_loader_det, optimizer_det, criterion_det)
    val_loss = evaluate_det(det_model, test_loader_det, criterion_det)
    print(f"Epoch {epoch+1}/{num_epochs_det}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
# 对检测模型保存训练参数
paddle.save(det_model.state_dict(), "det_model_final.pdparams")

# -------------------------------
# 7. 可视化部分（示例）
# -------------------------------
# 取测试集中一张图片，比较预测边框与真实边框
det_model.eval()
img_sample, bbox_sample = test_dataset_det[0]
img_np = img_sample.numpy().transpose(1, 2, 0)  # 转为HWC格式
pred_bbox = det_model(img_sample.unsqueeze(0)).numpy()[0]

# 绘制真实边框和预测边框
def draw_bbox(image, bbox, color):
    pts = bbox.reshape((-1, 2)).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    return image

img_draw = img_np.copy()
img_draw = draw_bbox(img_draw, np.array(bbox_sample), (0,255,0))      # 绿色：真实边框
img_draw = draw_bbox(img_draw, pred_bbox, (255,0,0))  # 蓝色：预测边框

plt.imshow(img_draw.astype(np.uint8))
plt.title("Detection: Green: GT, Blue: Pred")
plt.show()