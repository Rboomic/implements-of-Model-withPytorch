import torch
import torch.nn as tnn
import torch.utils.data as tud
import numpy as np

# 超参数定义

LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  #输出类别数

# 数据路径
TRAIN_FEATURES_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWX.npy'
TRAIN_LABELS_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWY.npy'
TEST_FEATURES_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWX_test.npy'
TEST_LABELS_PATH = 'D:/MakeThing/DaChuang/deeplearning/dataProcess/NEWY_test.npy'

# 超参数
BATCH_SIZE = 32  # 定义批次大小
INPUT_SIZE = (224, 224)  # 输入图像的空间尺寸
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
class MyDataset(tud.Dataset):
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

train_loader = tud.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = tud.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义卷积层
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


# 定义VGG16模型
class VGG16(tnn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(VGG16, self).__init__()

        # 输入尺寸调整：82x82x1
        self.layer1 = conv_layer(1, 64, 3, 1)
        self.layer2 = conv_layer(64, 64, 3, 1)
        self.pool1 = tnn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = conv_layer(64, 128, 3, 1)
        self.layer4 = conv_layer(128, 128, 3, 1)
        self.pool2 = tnn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5 = conv_layer(128, 256, 3, 1)
        self.layer6 = conv_layer(256, 256, 3, 1)
        self.layer7 = conv_layer(256, 256, 3, 1)
        self.pool3 = tnn.MaxPool2d(kernel_size=2, stride=2)

        self.layer8 = conv_layer(256, 512, 3, 1)
        self.layer9 = conv_layer(512, 512, 3, 1)
        self.layer10 = conv_layer(512, 512, 3, 1)
        self.pool4 = tnn.MaxPool2d(kernel_size=2, stride=2)

        self.layer11 = conv_layer(512, 512, 3, 1)
        self.layer12 = conv_layer(512, 512, 3, 1)
        self.layer13 = conv_layer(512, 512, 3, 1)
        self.pool5 = tnn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = tnn.Linear(512 * 7 * 7, 4096)  # 根据池化后的尺寸计算输入特征数
        self.fc2 = tnn.Linear(4096, 4096)
        self.fc3 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.pool1(out)

        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)

        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.pool3(out)

        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.pool4(out)

        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.pool5(out)

        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)

        return out


# 初始化模型、损失函数和优化器
model = VGG16(n_classes=N_CLASSES)
model.cuda()

criterion = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练模型
def train_model(model, loader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')


# 测试模型
def test_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')


# 开始训练和测试
train_model(model, train_loader, optimizer, criterion, EPOCH)
test_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'vgg16_model.pth')