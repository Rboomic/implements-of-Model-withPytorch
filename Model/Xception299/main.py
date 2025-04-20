import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import time
import os
from datetime import datetime

# 超参数定义
LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  # 输出类别数
BATCH_SIZE = 32  # 批次大小
INPUT_SIZE = (299, 299)  # 输入图像的空间尺寸
NUM_CHANNELS = 1  # 输入通道数

# 数据路径
TRAIN_FEATURES_PATH = 'D:/MakeThing/DaChuang/deeplearning/DataProcess2/NEWX.npy'
TRAIN_LABELS_PATH = 'D:/MakeThing/DaChuang/deeplearning/DataProcess2/NEWY.npy'
TEST_FEATURES_PATH = 'D:/MakeThing/DaChuang/deeplearning/DataProcess2/NEWX_test.npy'
TEST_LABELS_PATH = 'D:/MakeThing/DaChuang/deeplearning/DataProcess2/NEWY_test.npy'

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
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# 创建数据加载器
train_dataset = MyDataset(X_train_tensor, y_train_tensor)
test_dataset = MyDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义SeparableConv块
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels,
                               bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


# 定义Xception模型
class Xception(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super(Xception, self).__init__()

        # Entry flow
        self.entry_flow = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            self._block(64, 128, 2, 2),
            self._block(128, 256, 2, 2),
            self._block(256, 728, 2, 2)
        )

        # Middle flow
        self.middle_flow = nn.Sequential(
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1),
            self._block(728, 728, 3, 1)
        )

        # Exit flow
        self.exit_flow = nn.Sequential(
            self._block(728, 1024, 2, 2),
            SeparableConv(1024, 1536, kernel_size=3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),

            SeparableConv(1536, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(2048, num_classes)

    def _block(self, in_channels, out_channels, num_conv, stride):
        block = nn.Sequential()
        for i in range(num_conv):
            if i == 0:
                block.add_module(f'conv{i + 1}',
                                 nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3,
                                           stride=stride if i == 0 else 1, padding=1, bias=False))
            else:
                block.add_module(f'conv{i + 1}',
                                 nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            block.add_module(f'bn{i + 1}', nn.BatchNorm2d(out_channels))
            block.add_module(f'relu{i + 1}', nn.ReLU(inplace=True))
        return block

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


# 初始化模型、损失函数和优化器
model = Xception()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 定义训练函数
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# 定义测试函数
def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return epoch_loss, accuracy


# 定义评估函数
def evaluate(model, loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    return accuracy, report, matrix


# 设置设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 创建日志目录
log_dir = f'logs_{datetime.now().strftime("%Y%m%d%H%M%S")}'
os.makedirs(log_dir, exist_ok=True)

# 训练模型
best_accuracy = 0.0
train_history = []
test_history = []

for epoch in range(EPOCH):
    print(f'Epoch {epoch + 1}/{EPOCH}')
    print('-' * 10)

    # 记录开始时间
    start_time = time.time()

    # 训练
    train_loss = train(model, train_loader, criterion, optimizer)

    # 测试
    test_loss, test_accuracy = test(model, test_loader, criterion)

    # 计算F1 Score
    _, _, test_conf_matrix = evaluate(model, test_loader)
    test_f1 = test_conf_matrix.trace() / test_conf_matrix.sum()

    # 记录时间
    epoch_time = time.time() - start_time

    # 记录指标
    train_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'time': epoch_time
    })

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Epoch Time: {epoch_time:.2f}s')
    print()

    # 更新最佳准确率
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'accuracy': test_accuracy
        }, os.path.join(log_dir, 'best_model.pth'))
        print('Best model saved.')

# 在测试集上评估最佳模型
model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth'))['model_state_dict'])
test_accuracy, report, matrix = evaluate(model, test_loader)

# 保存训练和测试结果
train_df = pd.DataFrame(train_history)
test_results = {
    'final_accuracy': [test_accuracy],
    'final_f1': [matrix.trace() / matrix.sum()],
    'final_loss': [test_loss]
}
test_df = pd.DataFrame(test_results)

# 保存到CSV
train_df.to_csv(os.path.join(log_dir, 'train_history.csv'), index=False)
test_df.to_csv(os.path.join(log_dir, 'test_results.csv'), index=False)

# 保存分类报告和混淆矩阵
with open(os.path.join(log_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

np.save(os.path.join(log_dir, 'confusion_matrix.npy'), matrix)


# 保存训练集和测试集的预测结果
def get_predictions(model, loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_true, y_pred


# 获取训练集和测试集的预测结果
train_y_true, train_y_pred = get_predictions(model, train_loader)
test_y_true, test_y_pred = get_predictions(model, test_loader)

# 保存预测结果
np.savez(os.path.join(log_dir, 'train_predictions.npz'),
         y_true=train_y_true, y_pred=train_y_pred)
np.savez(os.path.join(log_dir, 'test_predictions.npz'),
         y_true=test_y_true, y_pred=test_y_pred)

print('All results saved successfully.')