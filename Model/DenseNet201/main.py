import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from collections import OrderedDict
import time
import psutil
import os
import pandas as pd

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
        dataset = torch.utils.data.TensorDataset(features, labels)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)


# 创建数据加载器
train_loader = MyDataset(X_train_tensor, y_train_tensor, BATCH_SIZE, shuffle=True)
test_loader = MyDataset(X_test_tensor, y_test_tensor, BATCH_SIZE, shuffle=False)


# 定义密集层
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


# 定义密集块
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


# 定义过渡层
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# 定义DenseNet201模型
class DenseNet201(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=9):
        super(DenseNet201, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(NUM_CHANNELS, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# 初始化模型、损失函数和优化器
model = DenseNet201(num_classes=N_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)


# 定义训练函数
def train(model, loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(output, 1)
        running_corrects += torch.sum(preds == target).item()
    end_time = time.time() - start_time
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc, end_time


# 定义测试函数
def test(model, loader):
    model.eval()
    start_time = time.time()
    test_loss = 0
    correct = 0
    targets = []
    preds = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            targets.extend(target.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    end_time = time.time() - start_time
    avg_loss = test_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    f1 = f1_score(targets, preds, average='weighted')
    return avg_loss, accuracy, f1, targets, preds, end_time


# 使用GPU训练，如果有的话
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 初始化记录数据
history = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    'test_f1': [],
    'train_time': [],
    'test_time': [],
    'memory_usage': []
}

best_accuracy = 0.0
best_f1 = 0.0

# 训练模型
for epoch in range(EPOCH):
    # 训练阶段
    start_train_time = time.time()
    train_loss, train_acc, train_time = train(model, train_loader, optimizer, epoch)
    end_train_time = time.time() - start_train_time

    # 测试阶段
    test_loss, test_acc, test_f1, test_targets, test_preds, test_time = test(model, test_loader)

    # 内存使用情况
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # 转换为MB

    # 记录数据
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    history['test_f1'].append(test_f1)
    history['train_time'].append(train_time)
    history['test_time'].append(test_time)
    history['memory_usage'].append(memory_usage)

    # 保存最佳模型
    if test_acc > best_accuracy or test_f1 > best_f1:
        best_accuracy = max(test_acc, best_accuracy)
        best_f1 = max(test_f1, best_f1)
        torch.save(model.state_dict(), 'best_densenet201_model.pth')

    # 打印进度
    print(f'Epoch {epoch + 1}/{EPOCH}')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}')
    print(f'Train Time: {train_time:.2f}s | Test Time: {test_time:.2f}s | Memory Usage: {memory_usage:.2f}MB')
    print('------------------------')

# 测试模型
final_test_loss, final_test_acc, final_test_f1, final_test_targets, final_test_preds, final_test_time = test(model,
                                                                                                             test_loader)

# 生成分类报告和混淆矩阵
final_classification_report = classification_report(final_test_targets, final_test_preds)
final_confusion_matrix = confusion_matrix(final_test_targets, final_test_preds)

# 将结果保存到DataFrame
df = pd.DataFrame(history)
df.to_excel('training_results.xlsx', index=False)

# 保存分类报告和混淆矩阵
with open('classification_report.txt', 'w') as f:
    f.write(final_classification_report)

with open('confusion_matrix.txt', 'w') as f:
    f.write(str(final_confusion_matrix))

print('All results have been saved to Excel and text files.')