import torch
import torch.nn as nn
import torch.utils.data as tud
import numpy as np
import time
import psutil
import os
from sklearn.metrics import f1_score
import pandas as pd

# 超参数定义
LEARNING_RATE = 0.01
EPOCH = 15
N_CLASSES = 9  # 输出类别数
BATCH_SIZE = 32
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

train_dataset = MyDataset(X_train_tensor, y_train_tensor)
test_dataset = MyDataset(X_test_tensor, y_test_tensor)

train_loader = tud.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = tud.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 50,101,152
class Bottleneck(nn.Module):
    expansion = 4  # 4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,  # 输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet50, self).__init__()
        block = Bottleneck
        layers = [3, 4, 6, 3]
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(NUM_CHANNELS, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_acc += torch.sum(preds == labels.data).item()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}')

    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'f1': epoch_f1,
        'preds': all_preds,
        'labels': all_labels
    }

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels.data).item()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_acc / len(val_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Validation Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}')

    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'f1': epoch_f1,
        'preds': all_preds,
        'labels': all_labels
    }

def main():
    # 初始化模型、优化器、损失函数和设备
    model = ResNet50(num_classes=N_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建保存目录
    save_dir = 'training_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化记录数据
    history = []

    # 开始训练
    start_time = time.time()
    best_val_acc = 0.0
    best_val_f1 = 0.0

    for epoch in range(EPOCH):
        print(f'Epoch {epoch+1}/{EPOCH}')
        print('-' * 10)

        # 记录epoch开始时间
        epoch_start = time.time()

        # 训练阶段
        train_stats = train(model, train_loader, criterion, optimizer, epoch, device)

        # 更新学习率
        scheduler.step()

        # 评估阶段
        val_stats = evaluate(model, test_loader, criterion, device)

        # 记录运行时间
        epoch_time = time.time() - epoch_start

        # 获取内存使用情况
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / (1024 * 1024)  # MB

        # 收集本epoch的数据
        history_epoch = {
            'epoch': epoch + 1,
            'train_loss': train_stats['loss'],
            'train_acc': train_stats['acc'],
            'train_f1': train_stats['f1'],
            'val_loss': val_stats['loss'],
            'val_acc': val_stats['acc'],
            'val_f1': val_stats['f1'],
            'time': epoch_time,
            'memory': mem_usage
        }
        history.append(history_epoch)

        # 保存最佳模型
        if val_stats['acc'] > best_val_acc or val_stats['f1'] > best_val_f1:
            if val_stats['f1'] > best_val_f1:
                best_val_acc = val_stats['acc']
                best_val_f1 = val_stats['f1']
                torch.save(model.state_dict(), os.path.join(save_dir, 'resnet50_best.pth'))
                print('Best model saved.')

    # 保存最后的模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'resnet50_last.pth'))

    # 将历史记录保存到CSV
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    # 保存训练集和测试集的分类结果
    train_preds = train_stats['preds']
    train_labels = train_stats['labels']
    np.save(os.path.join(save_dir, 'train_preds.npy'), train_preds)
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)

    test_preds = val_stats['preds']
    test_labels = val_stats['labels']
    np.save(os.path.join(save_dir, 'test_preds.npy'), test_preds)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')

if __name__ == '__main__':
    main()