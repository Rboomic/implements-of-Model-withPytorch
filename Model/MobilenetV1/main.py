import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

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

# 定义DSC结构：DW+PW操作
def BottleneckV1(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                  groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

# 定义MobileNetV1结构
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super(MobileNetV1, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            BottleneckV1(32, 64, stride=1),
            BottleneckV1(64, 128, stride=2),
            BottleneckV1(128, 128, stride=1),
            BottleneckV1(128, 256, stride=2),
            BottleneckV1(256, 256, stride=1),
            BottleneckV1(256, 512, stride=2),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 1024, stride=2),
            BottleneckV1(1024, 1024, stride=1),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

def train_model(model, device, train_loader, test_loader, criterion, optimizer, num_epochs=15):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    return model

def test_model(model, device, test_loader):
    model.eval()
    total_loss = 0
    total_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels).item()
    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = total_corrects / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}, Test Acc: {avg_acc:.4f}')
    return avg_loss, avg_acc

def evaluate_model(model, device, test_loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

def model_save(model, path='mobilenetv1.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def model_load(model, path='mobilenetv1.pth'):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 检测是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化模型
    model = MobileNetV1().to(device)

    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    model = train_model(model, device, train_loader, test_loader, criterion, optimizer, EPOCH)

    # 测试模型
    test_loss, test_acc = test_model(model, device, test_loader)

    # 评估模型
    evaluate_model(model, device, test_loader)

    # 保存模型
    model_save(model)