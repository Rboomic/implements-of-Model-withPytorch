import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import time
import psutil
import os
import pandas as pd
from sklearn.metrics import f1_score

# 超参数定义
LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  # 输出类别数
BATCH_SIZE = 32  # 批次大小
INPUT_SIZE = (224, 224)  # 输入图像的空间尺寸
NUM_CHANNELS = 1  # 输入通道数

# 数据路径
TRAIN_FEATURES_PATH = '../NEWX_train_large.npy'
TRAIN_LABELS_PATH = '../NEWY_train_large.npy'
TEST_FEATURES_PATH = '../NEWX_test_large.npy'
TEST_LABELS_PATH = '../NEWY_test_large.npy'

# 加载数据
X_train = np.load(TRAIN_FEATURES_PATH)
y_train = np.load(TRAIN_LABELS_PATH)
X_test = np.load(TEST_FEATURES_PATH)
y_test = np.load(TEST_LABELS_PATH)

# 数据预处理
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int64)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int64)

# 添加通道维度
X_train = X_train.reshape((-1, NUM_CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]))
X_test = X_test.reshape((-1, NUM_CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]))

# 转换为PyTorch Tensor
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

# 定义数据集类
class MyDataset(DataLoader):
    def __init__(self, features, labels, batch_size, shuffle=True):
        dataset = TensorDataset(features, labels)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

# 创建数据加载器
train_loader = MyDataset(X_train_tensor, y_train_tensor, BATCH_SIZE, shuffle=True)
test_loader = MyDataset(X_test_tensor, y_test_tensor, BATCH_SIZE, shuffle=False)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape: b, num_channels, h, w  -->  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ShuffleNetUnit(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnit, self).__init__()

        mid_channels = out_channels // 4
        self.stride = stride
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups

        self.g_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.g_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.g_conv1(x)
        out = channel_shuffle(out, groups=self.groups)
        out = self.dw_conv(out)
        out = self.g_conv2(out)
        short = self.shortcut(x)
        if self.stride == 2:
            out = F.relu(torch.cat([out, short], dim=1))
        else:
            out = F.relu(out + short)
        return out

class ShuffleNet(nn.Module):
    def __init__(self, groups, num_layers, num_channels, num_classes=N_CLASSES):
        super(ShuffleNet, self).__init__()

        self.groups = groups

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 阶段2
        self.stage2 = self.make_layers(24, num_channels[0], num_layers[0], groups)
        # 阶段3
        self.stage3 = self.make_layers(num_channels[0], num_channels[1], num_layers[1], groups)
        # 阶段4
        self.stage4 = self.make_layers(num_channels[1], num_channels[2], num_layers[2], groups)

        # 全局平均池化
        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)

        # 全连接层
        self.fc = nn.Linear(num_channels[2], num_classes)

    def make_layers(self, in_channels, out_channels, num_layers, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels - in_channels, 2, groups))
        in_channels = out_channels
        for i in range(num_layers - 1):
            layers.append(ShuffleNetUnit(in_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def ShuffleNet_g1(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [144, 288, 576]
    model = ShuffleNet(1, num_layers, num_channels, **kwargs)
    return model

def ShuffleNet_g2(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [200, 400, 800]
    model = ShuffleNet(2, num_layers, num_channels, **kwargs)
    return model

def ShuffleNet_g3(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [240, 480, 960]
    model = ShuffleNet(3, num_layers, num_channels, **kwargs)
    return model

def ShuffleNet_g4(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [272, 544, 1088]
    model = ShuffleNet(4, num_layers, num_channels, **kwargs)
    return model

def ShuffleNet_g8(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [384, 768, 1536]
    model = ShuffleNet(8, num_layers, num_channels, **kwargs)
    return model

def train_model(model, train_loader, test_loader, num_epochs=EPOCH, learning_rate=LEARNING_RATE):
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练和测试的损失及准确率
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_f1 = []
    test_f1 = []
    epoch_times = []
    mem_usage = []

    best_test_acc = 0
    best_test_f1 = 0
    best_model_params = None

    # 保存分类结果
    train_preds = []
    train_labels = []
    test_preds = []
    test_labels = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # 记录内存使用情况
        process = psutil.Process(os.getpid())
        mem_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

        model.train()
        running_loss = 0.0
        running_corrects = 0
        epoch_preds = []
        epoch_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

            # 收集分类结果
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 计算F1
        train_f1_epoch = f1_score(epoch_labels, epoch_preds, average='weighted')
        train_f1.append(train_f1_epoch)

        # 测试阶段
        model.eval()
        test_loss = 0
        corrects = 0
        test_epoch_preds = []
        test_epoch_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data).item()

                # 收集分类结果
                test_epoch_preds.extend(preds.cpu().numpy())
                test_epoch_labels.extend(labels.cpu().numpy())

        test_epoch_loss = test_loss / len(test_loader.dataset)
        test_epoch_acc = corrects / len(test_loader.dataset)
        test_losses.append(test_epoch_loss)
        test_accuracies.append(test_epoch_acc)

        # 计算F1
        test_f1_epoch = f1_score(test_epoch_labels, test_epoch_preds, average='weighted')
        test_f1.append(test_f1_epoch)

        # 记录时间
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # 更新最佳模型
        if test_epoch_acc > best_test_acc:
            best_test_acc = test_epoch_acc
            best_test_f1 = test_f1_epoch
            best_model_params = model.state_dict()

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train F1: {train_f1_epoch:.4f}')
        print(f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.4f}, Test F1: {test_f1_epoch:.4f}')
        print(f'Memory Usage: {mem_usage[-1]:.2f} MB')
        print(f'Time: {epoch_time:.2f}s')
        print('-' * 10)

    # 保存分类结果
    train_results = pd.DataFrame({
        'Prediction': train_preds,
        'Label': train_labels
    })
    test_results = pd.DataFrame({
        'Prediction': test_preds,
        'Label': test_labels
    })
    train_results.to_csv('train_classification_results.csv', index=False)
    test_results.to_csv('test_classification_results.csv', index=False)

    # 保存训练和测试数据
    metrics = pd.DataFrame({
        'Epoch': range(1, num_epochs+1),
        'Train Loss': train_losses,
        'Train Acc': train_accuracies,
        'Train F1': train_f1,
        'Test Loss': test_losses,
        'Test Acc': test_accuracies,
        'Test F1': test_f1,
        'Time (s)': epoch_times,
        'Memory (MB)': mem_usage
    })
    metrics.to_csv('training_metrics.csv', index=False)

    # 保存最佳模型
    torch.save(best_model_params, 'best_model.pth')

    return model, train_losses, train_accuracies, test_losses, test_accuracies

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_loss = 0
    corrects = 0
    criterion = nn.CrossEntropyLoss()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_epoch_loss = test_loss / len(test_loader.dataset)
    test_epoch_acc = corrects / len(test_loader.dataset)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    print(f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.4f}, Test F1: {test_f1:.4f}')

    # 保存测试分类结果
    test_results = pd.DataFrame({
        'Prediction': test_preds,
        'Label': test_labels
    })
    test_results.to_csv('test_classification_results.csv', index=False)

if __name__ == '__main__':
    # 初始化模型
    model = ShuffleNet_g1(num_classes=N_CLASSES)

    # 训练模型
    start_time = time.time()
    trained_model, train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, train_loader, test_loader)
    total_time = time.time() - start_time

    # 测试模型
    test_model(trained_model, test_loader)

    # 保存总运行时间
    with open('total_running_time.txt', 'w') as f:
        f.write(f'Total training time: {total_time:.2f}s')