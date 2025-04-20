import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

        self.conv1 = conv3x3(in_planes =in_planes,out_channels=32,stride=2, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32,  stride=1, padding=0)
        self.conv3 = conv3x3(in_planes=32, out_channels=64, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3,  stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=64, out_channels=64,  stride=1, padding=1)
        self.conv5 = conv1x1(in_planes =64,out_channels=80,  stride=1, padding=0)
        self.conv6 = conv3x3(in_planes=80, out_channels=192, stride=1, padding=0)
        self.conv7 = conv3x3(in_planes=192, out_channels=256,  stride=2, padding=0)

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
    def __init__(self, input ):
        super(Inception_ResNet_A, self).__init__()
        self.conv1 = conv1x1(in_planes =input,out_channels=32,stride=1, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32, stride=1, padding=1)
        self.line =  nn.Conv2d(96, 256, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):

        c1 = self.conv1(x)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c3 = self.conv1(x)
        # print("c3", c3.shape)
        c2_1 = self.conv2(c2)
        # print("c2_1", c2_1.shape)
        c3_1 = self.conv2(c3)
        # print("c3_1", c3_1.shape)
        c3_2 = self.conv2(c3_1)
        # print("c3_2", c3_2.shape)
        cat = torch.cat([c1, c2_1, c3_2],dim=1)#torch.Size([4, 96, 15, 15])
        # print("x",x.shape)

        line = self.line(cat)
        # print("line",line.shape)
        out =x+line
        out = self.relu(out)

        return out
class Reduction_A(nn.Module):
    def __init__(self, input,n=384,k=192,l=224,m=256):
        super(Reduction_A, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,  stride=2, padding=0)
        self.conv1 = conv3x3(in_planes=input, out_channels=n,stride=2,padding=0)
        self.conv2 = conv1x1(in_planes=input, out_channels=k,padding=1)
        self.conv3 = conv3x3(in_planes=k,  out_channels=l,padding=0)
        self.conv4 = conv3x3(in_planes=l,  out_channels=m,stride=2,padding=0)

    def forward(self, x):

        c1 = self.maxpool(x)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c3 = self.conv2(x)
        # print("c3", c3.shape)
        c3_1 = self.conv3(c3)
        # print("c3_1", c3_1.shape)
        c3_2 = self.conv4(c3_1)
        # print("c3_2", c3_2.shape)
        cat = torch.cat([c1, c2,c3_2], dim=1)

        return cat
class Inception_ResNet_B(nn.Module):
    def __init__(self, input):
        super(Inception_ResNet_B, self).__init__()
        self.conv1 = conv1x1(in_planes =input,out_channels=128,stride=1, padding=0)
        self.conv1x7 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(1,7), padding=(0,3))
        self.conv7x1 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(7,1), padding=(3,0))
        self.line = nn.Conv2d(256, 896, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):

        c1 = self.conv1(x)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c2_1 = self.conv1x7(c2)
        # print("c2_1", c2_1.shape)

        c2_1 = self.relu(c2_1)

        c2_2 = self.conv7x1(c2_1)
        # print("c2_2", c2_2.shape)

        c2_2 = self.relu(c2_2)

        cat = torch.cat([c1, c2_2], dim=1)
        line = self.line(cat)
        out =x+line

        out = self.relu(out)
        # print("out", out.shape)
        return out


class Reduction_B(nn.Module):
    def __init__(self, input):
        super(Reduction_B, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = conv1x1(in_planes=input,  out_channels=256, padding=1)
        self.conv2 = conv3x3(in_planes=256,  out_channels=384, stride=2, padding=0)
        self.conv3 = conv3x3(in_planes=256,  out_channels=256,stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=256,  out_channels=256,  padding=1)
        self.conv5 = conv3x3(in_planes=256,  out_channels=256, stride=2, padding=0)
    def forward(self, x):
        c1 = self.maxpool(x)
        # print("c1", c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c3 = self.conv1(x)
        # print("c3", c3.shape)
        c4 = self.conv1(x)
        # print("c4", c4.shape)
        c2_1 = self.conv2(c2)
        # print("cc2_1", c2_1.shape)
        c3_1 = self.conv3(c3)
        # print("c3_1", c3_1.shape)
        c4_1 = self.conv4(c4)
        # print("c4_1", c4_1.shape)
        c4_2 = self.conv5(c4_1)
        # print("c4_2", c4_2.shape)
        cat = torch.cat([c1, c2_1, c3_1,c4_2], dim=1)
        # print("cat", cat.shape)
        return cat

class Inception_ResNet_C(nn.Module):
    def __init__(self, input):
        super(Inception_ResNet_C, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=192, stride=1, padding=0)
        self.conv1x3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0,1))
        self.conv3x1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1,0))
        self.line = nn.Conv2d(384, 1792, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        # print("x", x.shape)
        # print("c1",c1.shape)
        c2 = self.conv1(x)
        # print("c2", c2.shape)
        c2_1 = self.conv1x3(c2)
        # print("c2_1", c2_1.shape)

        c2_1 = self.relu(c2_1)
        c2_2 = self.conv3x1(c2_1)
        # print("c2_2", c2_2.shape)

        c2_2 = self.relu(c2_2)
        cat = torch.cat([c1, c2_2], dim=1)
        # print("cat", cat.shape)
        line = self.line(cat)
        out = x+ line
        # print("out", out.shape)
        out = self.relu(out)

        return out


class Inception_ResNet(nn.Module):
    def __init__(self,classes=2):
        super(Inception_ResNet, self).__init__()
        blocks = []
        blocks.append(StemV1(in_planes=3))
        for i in range(5):
            blocks.append(Inception_ResNet_A(input=256))
        blocks.append(Reduction_A(input=256))
        for i in range(10):
            blocks.append(Inception_ResNet_B(input=896))
        blocks.append(Reduction_B(input=896))
        for i in range(10):
            blocks.append(Inception_ResNet_C(input=1792))

        self.features = nn.Sequential(*blocks)

        self.avepool = nn.AvgPool2d(kernel_size=3)

        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(1792, classes)

    def forward(self,x):
        x = self.features(x)
        # print("x",x.shape)
        x = self.avepool(x)
        # print("avepool", x.shape)
        x = self.dropout(x)
        # print("dropout", x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return  x

class Inception_ResNet(nn.Module):
    def __init__(self, classes=N_CLASSES):
        super(Inception_ResNet, self).__init__()
        blocks = []
        # 修改StemV1的in_planes为NUM_CHANNELS
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

        self.avepool = nn.AvgPool2d(kernel_size=3)

        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(1792, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avepool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型、优化器和损失函数
model = Inception_ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
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
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

# 测试模型
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
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')
    return all_preds, all_labels

# 训练模型
train_model(model, train_loader, optimizer, criterion, EPOCH)

# 测试模型
preds, labels = test_model(model, test_loader)

# 评估模型
print("\n分类报告:")
print(classification_report(labels, preds, target_names=[str(i) for i in range(N_CLASSES)]))
print("\n混淆矩阵:")
print(confusion_matrix(labels, preds))

# 保存模型
torch.save(model.state_dict(), "inception_resnet_v1.pth")
print("\n模型已保存为 inception_resnet_v1.pth")

