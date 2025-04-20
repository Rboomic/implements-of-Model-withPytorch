import torch.nn as nn
import torch
from torchsummary import summary
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 超参数定义
LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  # 输出类别数
BATCH_SIZE = 32  # 批次大小
INPUT_SIZE = (299, 299)  # 输入图像的空间尺寸
NUM_CHANNELS = 1  # 输入通道数

# 数据路径
TRAIN_FEATURES_PATH = '../NEWX_train_large299.npy'
TRAIN_LABELS_PATH = '../NEWY_train_large299.npy'
TEST_FEATURES_PATH = '../NEWX_test_large299.npy'
TEST_LABELS_PATH = '../NEWY_test_large299.npy'

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
class GoogLeNetV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNetV3, self).__init__()
        self.aux_logits = aux_logits
        # 3个3×3卷积替代7×7卷积
        self.conv1_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv1_2 = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv1_3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 80, kernel_size=3)
        self.conv3 = BasicConv2d(80, 192, kernel_size=3, stride=2)
        self.conv4 = BasicConv2d(192, 192, kernel_size=3, padding=1)

        self.inception3a = InceptionV3A(192, 64, 48, 64, 64, 96, 32)
        self.inception3b = InceptionV3A(256, 64, 48, 64, 64, 96, 64)
        self.inception3c = InceptionV3A(288, 64, 48, 64, 64, 96, 64)

        self.inception4a = InceptionV3D(288, 0, 384, 384, 64, 96, 0)
        self.inception4b = InceptionV3B(768, 192, 128, 192, 128, 192, 192)
        self.inception4c = InceptionV3B(768, 192, 160, 192, 160, 192, 192)
        self.inception4d = InceptionV3B(768, 192, 160, 192, 160, 192, 192)
        self.inception4e = InceptionV3D(768, 0, 384, 384, 64, 128, 0)

        if self.aux_logits == True:
            self.aux = InceptionAux(in_channels=768, out_channels=num_classes)

        self.inception5a = InceptionV3C(1280, 320, 384, 384, 448, 384, 192)
        self.inception5b = InceptionV3C(2048, 320, 384, 384, 448, 384, 192)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.conv1_1(x)
        # N x 32 x 149 x 149
        x = self.conv1_2(x)
        # N x 32 x 147 x 147
        x = self.conv1_3(x)
        #  N x 32 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.conv2(x)
        # N x 80 x 71 x 71
        x = self.conv3(x)
        # N x 192 x 35 x 35
        x = self.conv4(x)
        # N x 192 x 35 x 35
        x = self.inception3a(x)
        # N x 256 x 35 x 35
        x = self.inception3b(x)
        # N x 288 x 35 x 35
        x = self.inception3c(x)
        # N x 288 x 35x 35
        x = self.inception4a(x)
        # N x 768 x 17 x 17
        x = self.inception4b(x)
        # N x 768 x 17 x 17
        x = self.inception4c(x)
        # N x 768 x 17 x 17
        x = self.inception4d(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:    # eval model lose this layer
            aux = self.aux(x)
        # N x 768 x 17 x 17
        x = self.inception4e(x)
        # N x 1280 x 8 x 8
        x = self.inception5a(x)
        # N x 2048 x 8 x 8
        x = self.inception5b(x)
        # N x 2048 x 7 x 7
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000(num_classes)
        if self.training and self.aux_logits:  # 训练阶段使用
            return x, aux
        return x
    # 对模型的权重进行初始化操作
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# InceptionV3A:BasicConv2d+MaxPool2d
class InceptionV3A(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3A, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+3×3卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )
        # 1×1卷积++3×3卷积+3×3卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, padding=1)         # 保证输出大小等于输入大小
        )
        # 3×3池化+1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# InceptionV3B:BasicConv2d+MaxPool2d
class InceptionV3B(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3B, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+1×3卷积+3×1卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3, ch3x3, kernel_size=[3, 1], padding=[1, 0])   # 保证输出大小等于输入大小
        )
        # 1×1卷积+1×3卷积+3×1卷积+1×3卷积+3×1卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0])  # 保证输出大小等于输入大小
        )
        # 3×3池化+1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# InceptionV3C:BasicConv2d+MaxPool2d
class InceptionV3C(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3C, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+1×3卷积+3×1卷积
        self.branch2_0 = BasicConv2d(in_channels, ch3x3red, kernel_size=1)
        self.branch2_1 = BasicConv2d(ch3x3red, ch3x3, kernel_size=[1, 3], padding=[0, 1])
        self.branch2_2 = BasicConv2d(ch3x3red, ch3x3, kernel_size=[3, 1], padding=[1, 0])

        # 1×1卷积+3×3卷积+1×3卷积+3×1卷积
        self.branch3_0 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
        )
        self.branch3_1 = BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1])
        self.branch3_2 = BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0])

        # 3×3池化+1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2_0 = self.branch2_0(x)
        branch2 = torch.cat([self.branch2_1(branch2_0), self.branch2_2(branch2_0)], dim=1)
        branch3_0 = self.branch3_0(x)
        branch3 = torch.cat([self.branch3_1(branch3_0), self.branch3_2(branch3_0)], dim=1)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# InceptionV3D:BasicConv2d+MaxPool2d
class InceptionV3D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3D, self).__init__()
        # ch1x1:没有1×1卷积
        # 1×1卷积+3×3卷积,步长为2
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, stride=2)
        )
        # 1×1卷积+3×3卷积+3×3卷积,步长为2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),   # 保证输出大小等于输入大小
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, stride=2)
        )
        # 3×3池化,步长为2
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2))
        # pool_proj:池化层后不再接卷积层

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        # 拼接
        outputs = [branch1,branch2, branch3]
        return torch.cat(outputs, 1)

# 辅助分类器:AvgPool2d+BasicConv2d+Linear+dropout
class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.conv2 = BasicConv2d(in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(p=0.7)
        self.linear = nn.Linear(in_features=768, out_features=out_channels)
    def forward(self, x):
        # N x 768 x 17 x 17
        x = self.averagePool(x)
        # N x 768 x 5 x 5
        x = self.conv1(x)
        # N x 128 x 5 x 5
        x = self.conv2(x)
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        out = self.linear(self.dropout(x))
        # N x num_classes
        return out

# 卷积组: Conv2d+BN+ReLU
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GoogLeNetV3().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

# 训练模型
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    best_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            if model.training and model.aux_logits:
                outputs, aux_outputs = model(inputs)
                loss = criterion(outputs, labels) + 0.3 * criterion(aux_outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # 更新最佳准确率
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')

    print(f'Best training accuracy: {best_acc:.4f}')


# 测试模型
def test(model, test_loader):
    model.eval()
    total_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
    accuracy = total_corrects.double() / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


# 定义评估指标
def evaluate(model, test_loader, device):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
        f1_score
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred, average="weighted"):.4f}')
    print(f'Recall: {recall_score(y_true, y_pred, average="weighted"):.4f}')
    print(f'F1 Score: {f1_score(y_true, y_pred, average="weighted"):.4f}')


# 开始训练和测试
if __name__ == '__main__':
    # 训练模型
    train(model, train_loader, optimizer, criterion, EPOCH)

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy = test(model, test_loader)

    # 评估模型
    evaluate(model, test_loader, device)

