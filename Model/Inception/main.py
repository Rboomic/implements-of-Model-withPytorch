import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# 超参数定义
LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  # 输出类别数
BATCH_SIZE = 32  # 批次大小
INPUT_SIZE = (224, 224)  # 输入图像的空间尺寸
NUM_CHANNELS = 1  # 输入通道数

# 数据路径
TRAIN_FEATURES_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWX.npy'
TRAIN_LABELS_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWY.npy'
TEST_FEATURES_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWX_test.npy'
TEST_LABELS_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWY_test.npy'

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
# 假设原始数据形状为 (num_samples, height, width)
# 我们需要将其转换为 (num_samples, channels, height, width)
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


# 定义层初始化函数
def layer_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # 使用kaiming初始化，适用于ReLU激活函数
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        # 初始化BatchNorm层的权重为1，偏置为0
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# 定义LRN层
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x / div
        return x


# 定义Inception模块
class InceptionBase(nn.Module):
    def __init__(self, input_size, config):
        super(InceptionBase, self).__init__()
        self.depth_dim = 1  # 通道维度

        # 1x1卷积
        self.conv1 = nn.Conv2d(input_size, config[0][0], kernel_size=1, stride=1, padding=0)

        # 3x3瓶颈卷积
        self.conv3_1 = nn.Conv2d(input_size, config[1][0], kernel_size=1, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1)

        # 5x5瓶颈卷积
        self.conv5_1 = nn.Conv2d(input_size, config[2][0], kernel_size=1, stride=1, padding=0)
        self.conv5_5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2)

        # 最大池化+1x1卷积
        self.max_pool = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        self.conv_max_1 = nn.Conv2d(input_size, config[3][1], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 1x1分支
        output1 = torch.relu(self.conv1(x))

        # 3x3分支
        output2 = torch.relu(self.conv3_1(x))
        output2 = torch.relu(self.conv3_3(output2))

        # 5x5分支
        output3 = torch.relu(self.conv5_1(x))
        output3 = torch.relu(self.conv5_5(output3))

        # 池化分支
        output4 = self.max_pool(x)
        output4 = torch.relu(self.conv_max_1(output4))

        # 拼接所有分支
        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)


# 定义Inception v1模型
class InceptionV1(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super(InceptionV1, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)

        # 卷积层2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # 卷积层3
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn3 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception模块
        self.inception3a = InceptionBase(192, [[64], [96, 128], [16, 32], [3, 32]])
        self.inception3b = InceptionBase(256, [[128], [128, 192], [32, 96], [3, 64]])
        self.max_pool_inc3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception4a = InceptionBase(480, [[192], [96, 204], [16, 48], [3, 64]])
        self.inception4b = InceptionBase(508, [[160], [112, 224], [24, 64], [3, 64]])
        self.inception4c = InceptionBase(512, [[128], [128, 256], [24, 64], [3, 64]])
        self.inception4d = InceptionBase(512, [[112], [144, 288], [32, 64], [3, 64]])
        self.inception4e = InceptionBase(528, [[256], [160, 320], [32, 128], [3, 128]])
        self.max_pool_inc4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBase(832, [[256], [160, 320], [48, 128], [3, 128]])
        self.inception5b = InceptionBase(832, [[384], [192, 384], [48, 128], [3, 128]])
        self.avg_pool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # 分类器
        self.fc = nn.Linear(1024, num_classes)

        # 应用层初始化
        self.apply(layer_init)

    def forward(self, x):
        # 初始卷积和池化
        x = torch.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.lrn1(x)

        # 卷积层2
        x = torch.relu(self.conv2(x))

        # 卷积层3
        x = torch.relu(self.conv3(x))
        x = self.max_pool3(self.lrn3(x))

        # Inception模块3a和3b
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool_inc3(x)

        # Inception模块4a到4e
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.max_pool_inc4(x)

        # Inception模块5a和5b
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool5(x)

        # 展平并通过分类器
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x


# 定义训练函数
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}')


# 定义测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    avg_loss = test_loss / len(test_loader)
    print(
        f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * accuracy:.0f}%)')
    return accuracy


# 定义评估函数
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy


# 主函数
def main():
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化模型和优化器
    model = InceptionV1(num_classes=N_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)

    # 训练循环
    best_accuracy = 0.0
    for epoch in range(EPOCH):
        print(f'Epoch {epoch + 1}/{EPOCH}')
        print('---------------------------')
        train(model, train_loader, optimizer, criterion, epoch, device)
        accuracy = test(model, test_loader, criterion, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'inception_v1_trained.pth')
            print('保存最佳模型')
        print('---------------------------')

    print(f'训练完成，最佳准确率：{best_accuracy:.2f}%')


if __name__ == '__main__':
    main()