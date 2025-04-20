import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import pandas as pd
import time
import psutil
import os

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
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 创建数据加载器
train_dataset = MyDataset(X_train_tensor, y_train_tensor)
test_dataset = MyDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        e1 = self.expand1x1_activation(self.expand1x1(x))
        e2 = self.expand3x3_activation(self.expand3x3(x))
        out = torch.cat([e1, e2], 1)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(NUM_CHANNELS, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(NUM_CHANNELS, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        out = torch.flatten(x, 1)
        return out


def _squeezenet(version, **kwargs):
    model = SqueezeNet(version, **kwargs)
    return model


def squeezenet1_0(**kwargs):
    return _squeezenet('1_0', **kwargs)


def squeezenet1_1(**kwargs):
    return _squeezenet('1_1', **kwargs)


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=15, device='cuda'):
    model.train()
    best_acc = 0
    # 初始化记录数据
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': [],
        'train_time': [],
        'memory_usage': []
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        y_pred_train = []
        y_true_train = []

        epoch_start = time.time()

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

            # Collect predictions for F1 score
            y_pred_train.extend(preds.cpu().numpy())
            y_true_train.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # Calculate F1 score for training
        from sklearn.metrics import f1_score
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro')

        # Evaluate on test set
        test_loss, test_acc, test_f1, y_pred_test, y_true_test = evaluate_model(model, test_loader, criterion, device)

        # Record memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB

        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_f1'].append(train_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        history['train_time'].append(time.time() - epoch_start)
        history['memory_usage'].append(memory_usage)

        # Save classification results
        pd.DataFrame({
            'y_true': y_true_train,
            'y_pred': y_pred_train
        }, index=[epoch] * len(y_true_train)).to_csv(f'train_results_epoch_{epoch + 1}.csv', index=False)

        pd.DataFrame({
            'y_true': y_true_test,
            'y_pred': y_pred_test
        }, index=[epoch] * len(y_true_test)).to_csv(f'test_results_epoch_{epoch + 1}.csv', index=False)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'squeezenet_best.pth')
            print('Best model saved.')

    total_time = time.time() - start_time
    # Save history to CSV
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    print(f'Total training time: {total_time:.2f} seconds')
    print(f'Best accuracy on test set: {best_acc:.4f}')


def evaluate_model(model, test_loader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    y_pred_test = []
    y_true_test = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

            y_pred_test.extend(preds.cpu().numpy())
            y_true_test.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects / len(test_loader.dataset)
    from sklearn.metrics import f1_score
    test_f1 = f1_score(y_true_test, y_pred_test, average='macro')

    return epoch_loss, epoch_acc, test_f1, y_pred_test, y_true_test


if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化模型
    model = squeezenet1_0(num_classes=N_CLASSES)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=EPOCH, device=device)

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load('squeezenet_best.pth'))
    test_loss, test_acc, test_f1, _, _ = evaluate_model(model, test_loader, criterion, device)
    print(f'Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}, Final Test F1: {test_f1:.4f}')