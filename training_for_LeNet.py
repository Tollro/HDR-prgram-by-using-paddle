# -*- coding: utf-8 -*-
"""
MNIST手写数字识别模型训练(飞桨版)
适用于新手的极简教程 
"""

import paddle
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import Normalize
from paddle.vision.models import LeNet
import matplotlib.pyplot as plt

# #################### 配置参数 ####################
BATCH_SIZE = 128      # 每次训练输入的图片数量（越大训练越快，但需要更多内存）
EPOCHS = 8           # 整个数据集遍历训练的次数
LR = 0.001           # 学习率（控制参数调整速度，太小训练慢，太大会震荡）
MODEL_SAVE_PATH = './mnist_model'  # 模型保存路径

# #################### 数据准备 ####################
def load_data():
    """
    加载MNIST数据集并进行预处理
    返回: 训练集和测试集的数据加载器
    """
    # 数据归一化：将像素值从[0,255]缩放到[-1,1]
    transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')

    # 下载并加载训练集
    train_dataset = MNIST(mode='train', transform=transform)
    #创建数据加载器，自动分批和打乱顺序
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers = 2,
        shuffle=True  # 打乱数据顺序，防止模型记忆顺序
    )

    # 加载测试集（验证模型效果）
    test_dataset = MNIST(mode='test', transform=transform)
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    return train_loader, test_loader

# #################### 模型定义 ####################
def create_model():
    """
    创建LeNet-5模型结构
    返回: 配置好的模型实例
    """
    # LeNet是经典的CNN结构，适合处理图像
    model = LeNet(num_classes=10)  # num_classes=10表示10个数字（0-9）
    
    # 打印模型结构（可选）
    paddle.summary(model, (1, 1, 28, 28))  # 输入形状：[批次, 通道, 高, 宽]
    return model

# #################### 训练过程 ####################
def train_model(model, train_loader, test_loader):
    """
    执行模型训练
    参数:
        model: 创建好的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 设置模型为训练模式（启用Dropout等训练专用层）
    model.train()

    #paddle.device.set_device('gpu:1')

# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
    # model.prepare(
    #     optimizer=paddle.optimizer.Adam(
    #         learning_rate=0.001, parameters=model.parameters()
    #     ),
    #     loss=paddle.nn.CrossEntropyLoss(),
    #     metrics=paddle.metric.Accuracy(),
    # )

    # 定义优化器和损失函数
    optimizer = paddle.optimizer.Adam(
         learning_rate=LR,
         parameters=model.parameters()  # 需要优化的参数
     )
    loss_fn = paddle.nn.CrossEntropyLoss()

    # 记录训练过程中的指标
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    #model.fit(train_loader, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # 开始训练循环
    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        total_loss = 0
        correct = 0
        for batch_id, data in enumerate(train_loader()):
            # 获取数据和标签
            x_data = data[0]  # 图像数据，形状为[BATCH_SIZE, 1, 28, 28]
            y_data = data[1]  # 标签，形状为[BATCH_SIZE, 1]

            # 前向传播（计算预测结果）
            predicts = model(x_data)

            # 计算损失
            loss = loss_fn(predicts, y_data)
            total_loss += loss.numpy()

            # 计算准确率
            correct += paddle.metric.accuracy(predicts, y_data).numpy() * BATCH_SIZE

            # 反向传播（计算梯度）
            loss.backward()

            # 更新参数
            optimizer.step()
            optimizer.clear_grad()

            # 每100个batch打印进度
            if batch_id % 100 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_id}, Loss: {loss.numpy():.4f}')

        # 计算本epoch的平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / len(train_loader.dataset)
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)

        # ---------- 验证阶段 ----------
        model.eval()  # 切换为评估模式
        val_loss, val_acc = evaluate_model(model, test_loader, loss_fn)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        model.train()  # 切换回训练模式

        # 打印本epoch结果
        print(f'\nEpoch {epoch+1} 训练结果:')
        print(f'训练损失: {avg_loss:.4f} | 训练准确率: {avg_acc:.4f}')
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}\n')


    # 用 evaluate 在测试集上对模型进行验证
    # eval_result = model.evaluate(test_loader, verbose=1)
    # print(eval_result)

    # 保存模型
    paddle.save(model.state_dict(), MODEL_SAVE_PATH + '.pdparams')
    paddle.save(optimizer.state_dict(), MODEL_SAVE_PATH + '.pdopt')
    print(f'模型已保存至: {MODEL_SAVE_PATH}')

    # # 用 predict 在测试集上对模型进行推理
    # test_result = model.predict(test_loader)

    # # 由于模型是单一输出，test_result的形状为[1, 10000]，10000是测试数据集的数据量。这里打印第一个数据的结果，这个数组表示每个数字的预测概率
    # print(len(test_result))
    # print(test_result[0][0])

    # # 从测试集中取出一张图片
    # img, label = test_loader[0]

    # # 打印推理结果，这里的argmax函数用于取出预测值中概率最高的一个的下标，作为预测标签
    # pred_label = test_result[0][0].argmax()
    # print("true label: {}, pred label: {}".format(label[0], pred_label))

    # # 使用matplotlib库，可视化图片
    # plt.imshow(img[0])


    return history

# #################### 模型评估 ####################
def evaluate_model(model, data_loader, loss_fn):
    """
    评估模型在数据集上的表现
    返回: 平均损失, 准确率
    """
    total_loss = 0
    correct = 0
    
    with paddle.no_grad():  # 不计算梯度，节省内存
        for data in data_loader():
            x_data = data[0]
            y_data = data[1]
            
            predicts = model(x_data)
            loss = loss_fn(predicts, y_data)
            total_loss += loss.numpy()
            correct += paddle.metric.accuracy(predicts, y_data).numpy() * BATCH_SIZE
    
    avg_loss = total_loss / len(data_loader)
    avg_acc = correct / len(data_loader.dataset)
    return avg_loss, avg_acc

# #################### 可视化训练结果 ####################
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# #################### 主程序 ####################
if __name__ == '__main__':
    # 1. 加载数据
    train_loader, test_loader = load_data()
    print('数据加载完成')

    # 2. 创建模型
    model = create_model()
    print('模型创建完成')

    # 3. 训练模型
    history = train_model(model, train_loader, test_loader)

    # 4. 可视化训练过程
    plot_history(history)