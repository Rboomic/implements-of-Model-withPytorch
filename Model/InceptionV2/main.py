import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

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

# 定义InceptionV2模型
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

class InceptionV2A(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV2A, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+3×3卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        # 1×1卷积+3×3卷积+3×3卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, padding=1)
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
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionV2B(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV2B, self).__init__()
        # 1×1卷积+3×3卷积,步长为2
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, stride=2, padding=1)
        )
        # 1×1卷积+3×3卷积+3×3卷积,步长为2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, stride=2, padding=1)
        )
        # 3×3池化,步长为2
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)

class GoogLeNetV2(nn.Module):
    def __init__(self, num_classes=N_CLASSES, aux_logits=True, init_weights=False):
        super(GoogLeNetV2, self).__init__()
        self.aux_logits = aux_logits

        # 修改conv1的输入通道数为1
        self.conv1 = BasicConv2d(NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = InceptionV2A(192, 64, 64, 64, 64, 96, 32)
        self.inception3b = InceptionV2A(256, 64, 64, 96, 64, 96, 64)
        self.inception3c = InceptionV2B(320, 0, 128, 160, 64, 96, 0)

        self.inception4a = InceptionV2A(576, 224, 64, 96, 96, 128, 128)
        self.inception4b = InceptionV2A(576, 192, 96, 128, 96, 128, 128)
        self.inception4c = InceptionV2A(576, 160, 128, 160, 128, 128, 128)
        self.inception4d = InceptionV2A(576, 96, 128, 192, 160, 160, 128)
        self.inception4e = InceptionV2B(576, 0, 128, 192, 192, 256, 0)

        self.inception5a = InceptionV2A(1024, 352, 192, 320, 160, 224, 128)
        self.inception5b = InceptionV2A(1024, 352, 192, 320, 160, 224, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 定义设备（GPU或CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
model = GoogLeNetV2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    best_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
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
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f'Best training accuracy: {best_acc:.4f}')

# 测试模型
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# 计算模型评估指标
def evaluate_model(model, test_loader):
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
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    # 模块列表
    summary(model, input_size=(NUM_CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]), device='cuda')
    train_model(model, train_loader, optimizer, criterion, EPOCH)
    model.load_state_dict(torch.load('best_model.pth'))
    test_model(model, test_loader, criterion)
    evaluate_model(model, test_loader)