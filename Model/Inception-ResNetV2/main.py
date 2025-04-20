import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
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


# 模型定义部分（保持不变）
class conv3x3(nn.Module):
    def __init__(self, in_planes, out_channels, stride=1, padding=0):
        super(conv3x3, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_planes, out_channels, kernel_size=3, stride=stride, padding=padding),  # 卷积核为3x3
            nn.BatchNorm2d(out_channels),  # BN层，防止过拟合以及梯度爆炸
            nn.ReLU()  # 激活函数
        )

    def forward(self, input):
        return self.conv3x3(input)


class conv1x1(nn.Module):
    def __init__(self, in_planes, out_channels, stride=1, padding=0):
        super(conv1x1, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_planes, out_channels, kernel_size=1, stride=stride, padding=padding),  # 卷积核为1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv1x1(input)


class StemV1(nn.Module):
    def __init__(self, in_planes):
        super(StemV1, self).__init__()

        self.conv1 = conv3x3(in_planes=in_planes, out_channels=32, stride=2, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32, stride=1, padding=0)
        self.conv3 = conv3x3(in_planes=32, out_channels=64, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=64, out_channels=64, stride=1, padding=1)
        self.conv5 = conv1x1(in_planes=64, out_channels=80, stride=1, padding=0)
        self.conv6 = conv3x3(in_planes=80, out_channels=192, stride=1, padding=0)
        self.conv7 = conv3x3(in_planes=192, out_channels=256, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x


class Inception_ResNet_A(nn.Module):
    def __init__(self, input):
        super(Inception_ResNet_A, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=32, stride=1, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32, stride=1, padding=1)
        self.line = nn.Conv2d(96, 256, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv1(x)
        c3 = self.conv1(x)
        c2_1 = self.conv2(c2)
        c3_1 = self.conv2(c3)
        c3_2 = self.conv2(c3_1)
        cat = torch.cat([c1, c2_1, c3_2], dim=1)
        line = self.line(cat)
        out = x + line
        out = self.relu(out)
        return out


class Reduction_A(nn.Module):
    def __init__(self, input, n=384, k=192, l=224, m=256):
        super(Reduction_A, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv1 = conv3x3(in_planes=input, out_channels=n, stride=2, padding=0)
        self.conv2 = conv1x1(in_planes=input, out_channels=k, padding=1)
        self.conv3 = conv3x3(in_planes=k, out_channels=l, padding=0)
        self.conv4 = conv3x3(in_planes=l, out_channels=m, stride=2, padding=0)

    def forward(self, x):
        c1 = self.maxpool(x)
        c2 = self.conv1(x)
        c3 = self.conv2(x)
        c3_1 = self.conv3(c3)
        c3_2 = self.conv4(c3_1)
        cat = torch.cat([c1, c2, c3_2], dim=1)
        return cat


class Inception_ResNet_B(nn.Module):
    def __init__(self, input):
        super(Inception_ResNet_B, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=128, stride=1, padding=0)
        self.conv1x7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7), padding=(0, 3))
        self.conv7x1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 1), padding=(3, 0))
        self.line = nn.Conv2d(256, 896, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv1(x)
        c2_1 = self.conv1x7(c2)
        c2_1 = self.relu(c2_1)
        c2_2 = self.conv7x1(c2_1)
        c2_2 = self.relu(c2_2)
        cat = torch.cat([c1, c2_2], dim=1)
        line = self.line(cat)
        out = x + line
        out = self.relu(out)
        return out


class Reduction_B(nn.Module):
    def __init__(self, input):
        super(Reduction_B, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = conv1x1(in_planes=input, out_channels=256, padding=1)
        self.conv2 = conv3x3(in_planes=256, out_channels=384, stride=2, padding=0)
        self.conv3 = conv3x3(in_planes=256, out_channels=256, stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=256, out_channels=256, padding=1)
        self.conv5 = conv3x3(in_planes=256, out_channels=256, stride=2, padding=0)

    def forward(self, x):
        c1 = self.maxpool(x)
        c2 = self.conv1(x)
        c3 = self.conv1(x)
        c4 = self.conv1(x)
        c2_1 = self.conv2(c2)
        c3_1 = self.conv3(c3)
        c4_1 = self.conv4(c4)
        c4_2 = self.conv5(c4_1)
        cat = torch.cat([c1, c2_1, c3_1, c4_2], dim=1)
        return cat


class Inception_ResNet_C(nn.Module):
    def __init__(self, input):
        super(Inception_ResNet_C, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=192, stride=1, padding=0)
        self.conv1x3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0, 1))
        self.conv3x1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1, 0))
        self.line = nn.Conv2d(384, 1792, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv1(x)
        c2_1 = self.conv1x3(c2)
        c2_1 = self.relu(c2_1)
        c2_2 = self.conv3x1(c2_1)
        c2_2 = self.relu(c2_2)
        cat = torch.cat([c1, c2_2], dim=1)
        line = self.line(cat)
        out = x + line
        out = self.relu(out)
        return out


class Inception_ResNet(nn.Module):
    def __init__(self, classes=N_CLASSES):
        super(Inception_ResNet, self).__init__()
        blocks = []
        blocks.append(StemV1(in_planes=NUM_CHANNELS))
        for i in range(5):
            blocks.append(Inception_ResNet_A(input=256))
        blocks.append(Reduction_A(input=256))
        for i in range(10):
            blocks.append(Inception_ResNet_B(input=896))
        blocks.append(Reduction_B(input=896))
        for i in range(5):
            blocks.append(Inception_ResNet_C(input=1792))

        self.features = nn.Sequential(*blocks)

        self.avepool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(1792 * 3 * 3, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avepool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# 初始化设备、模型、优化器和损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Inception_ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 创建保存目录
SAVE_DIR = "training_logs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 初始化日志数据
log_data = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "train_f1": [],
    "train_precision": [],
    "train_recall": [],
    "test_loss": [],
    "test_acc": [],
    "test_f1": [],
    "test_precision": [],
    "test_recall": [],
    "time": [],
    "memory": []
}


# 定义训练函数
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    best_test_acc = 0
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_corrects = 0
        all_preds_train = []
        all_labels_train = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted', labels=np.unique(all_labels_train))
        train_precision = precision_score(all_labels_train, all_preds_train, average='weighted',
                                          labels=np.unique(all_labels_train))
        train_recall = recall_score(all_labels_train, all_preds_train, average='weighted',
                                    labels=np.unique(all_labels_train))
        log_data["train_loss"].append(epoch_loss)
        log_data["train_acc"].append(epoch_acc)
        log_data["train_f1"].append(train_f1)
        log_data["train_precision"].append(train_precision)
        log_data["train_recall"].append(train_recall)
        epoch_time = time.time() - epoch_start_time
        log_data["time"].append(epoch_time)
        # 记录内存使用情况（当前进程的内存）
        process = psutil.Process(os.getpid())
        log_data["memory"].append(process.memory_info().rss / 1024 / 1024)  # 转换为MB
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

        # 测试模型
        model.eval()
        test_loss = 0
        correct = 0
        all_preds_test = []
        all_labels_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                all_preds_test.extend(preds.cpu().numpy())
                all_labels_test.extend(labels.cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_acc = correct.double() / len(test_loader.dataset)
        test_f1 = f1_score(all_labels_test, all_preds_test, average='weighted', labels=np.unique(all_labels_test))
        test_precision = precision_score(all_labels_test, all_preds_test, average='weighted',
                                         labels=np.unique(all_labels_test))
        test_recall = recall_score(all_labels_test, all_preds_test, average='weighted',
                                   labels=np.unique(all_labels_test))
        log_data["test_loss"].append(test_loss)
        log_data["test_acc"].append(test_acc)
        log_data["test_f1"].append(test_f1)
        log_data["test_precision"].append(test_precision)
        log_data["test_recall"].append(test_recall)
        log_data["epoch"].append(epoch + 1)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        # 更新最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print("Best model updated and saved.")
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return log_data


# 定义测试函数
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = correct.double() / len(test_loader.dataset)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', labels=np.unique(all_labels))
    test_precision = precision_score(all_labels, all_preds, average='weighted', labels=np.unique(all_labels))
    test_recall = recall_score(all_labels, all_preds, average='weighted', labels=np.unique(all_labels))
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    return test_loss, test_acc, test_f1, test_precision, test_recall, all_preds


# 训练模型并收集日志数据
log_data = train_model(model, train_loader, optimizer, criterion, EPOCH)

# 最后一次测试
test_loss, test_acc, test_f1, test_precision, test_recall, preds = test_model(model, test_loader)

# 保存日志到DataFrame
df = pd.DataFrame(log_data)
# 保存到Excel
df.to_excel(os.path.join(SAVE_DIR, "training_log.xlsx"), index=False)
# 保存到CSV
df.to_csv(os.path.join(SAVE_DIR, "training_log.csv"), index=False)

print("\n分类报告:")
print(classification_report(y_test, preds))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, preds))

# 保存最终模型
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
print("\n模型已保存为 final_model.pth")