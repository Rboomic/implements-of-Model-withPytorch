import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as tud

# 超参数定义
LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  # 输出类别数

# 数据路径
TRAIN_FEATURES_PATH = '../NEWX_train_large416.npy'
TRAIN_LABELS_PATH = '../NEWY_train_large416.npy'
TEST_FEATURES_PATH = '../NEWX_test_large416.npy'
TEST_LABELS_PATH = '../NEWY_test_large416.npy'

# 超参数
BATCH_SIZE = 32  # 定义批次大小
INPUT_SIZE = (416, 416)  # 输入图像的空间尺寸
NUM_CHANNELS = 1  # 输入通道数

# 加载数据
X_train = np.load(TRAIN_FEATURES_PATH)
y_train = np.load(TRAIN_LABELS_PATH)
X_test = np.load(TEST_FEATURES_PATH)
y_test = np.load(TEST_LABELS_PATH)

# 确保数据类型正确，X为float，y为long
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int64)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int64)

# 将数据添加通道维度，以适应模型的输入要求
X_train = X_train.reshape((-1, NUM_CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]))
X_test = X_test.reshape((-1, NUM_CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]))

# 转换为PyTorch Tensor
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label


# 创建数据集和数据加载器
train_dataset = MyDataset(X_train_tensor, y_train_tensor)
test_dataset = MyDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义DarkNet-19模型
class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        print("Initializing the darknet19 network ......")

        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(1, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 1024
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        self.conv_7 = nn.Conv2d(1024, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        x = self.conv_7(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# 初始化模型、损失函数和优化器
model = DarkNet_19()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 定义训练函数
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / len(train_loader.dataset)

    print(f'Epoch [{epoch + 1}/{EPOCH}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc


# 定义测试函数
def test_model(model, test_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects / len(test_loader.dataset)

    print(f'Epoch [{epoch + 1}/{EPOCH}] Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc, all_preds, all_labels


# 定义评估函数
def evaluate_model(preds, labels, num_classes):
    class_names = [str(i) for i in range(num_classes)]

    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        confusion_matrix[p, l] += 1

    # 计算精确率、召回率和F1分数
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

    # 计算总体准确率
    total_acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    # 打印评估结果
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("\nPrecision:", precision)
    print("\nRecall:", recall)
    print("\nF1 Score:", f1)
    print("\nTotal Accuracy:", total_acc)


# 训练和测试模型
best_acc = 0.0

for epoch in range(EPOCH):
    # 训练
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, epoch)

    # 测试
    test_loss, test_acc, all_preds, all_labels = test_model(model, test_loader, criterion, epoch)

    # 更新最佳准确率
    if test_acc > best_acc:
        best_acc = test_acc
        # 保存模型
        torch.save(model.state_dict(), 'darknet19_best.pth')
        print(f'Epoch [{epoch + 1}/{EPOCH}] Best Accuracy Updated: {best_acc:.4f}')

    # 评估模型
    evaluate_model(all_preds, all_labels, N_CLASSES)

print(f'Training Complete! Best Accuracy: {best_acc:.4f}')