import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as tud
from typing import Callable, Optional, List
from functools import partial
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

# 数据加载和预处理
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


# 定义自定义数据集
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


# 模型定义
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    :param ch: 输入特征矩阵的channel
    :param divisor: 基数
    :param min_ch: 最小通道数
    """
    if min_ch is None:
        min_ch = divisor
    #   将ch调整到距离8最近的整数倍
    #   int(ch + divisor / 2) // divisor 向上取整
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    #   确保向下取整时不会减少超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#   定义 卷积-BN-激活函数 联合操作
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 #  BN层
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 #  激活函数
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


#   SE模块
class SqueezeExcitaion(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitaion, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


#   定义V3的Config文件
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 #  阿尔法参数
                 width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


#   V3 倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Optional[Callable[..., nn.Module]]):
        super(InvertedResidual, self).__init__()

        #   判断步幅是否正确
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        #   初始化 block 为 Identity 模块，确保即使在没有需要额外操作的情况下，
        #   self.block 仍是一个有效的 PyTorch 模块，可以被调用。
        #   这样做可以防止在前向传播中出现 AttributeError。
        self.block = nn.Identity()  # 或者 self.block = nn.Sequential()

        #   判断是否使用残差连接
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        #   expand
        #   判断是否需要升维操作
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

            # depthwise
            layers.append(ConvBNActivation(cnf.expanded_c,
                                           cnf.expanded_c,
                                           kernel_size=cnf.kernel,
                                           stride=cnf.stride,
                                           groups=cnf.expanded_c,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))
            #   判断是否使用SE结构
            if cnf.use_se:
                layers.append(SqueezeExcitaion(cnf.expanded_c))

            #   project
            layers.append(ConvBNActivation(cnf.expanded_c,
                                           cnf.out_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

            self.block = nn.Sequential(*layers)
            self.out_channel = cnf.out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(1,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# 定义模型
inverted_residual_setting = [
    InvertedResidualConfig(
        input_c=16,
        kernel=3,
        expanded_c=64,
        out_c=16,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=16,
        kernel=3,
        expanded_c=72,
        out_c=24,
        use_se=False,
        activation="HS",
        stride=2,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=24,
        kernel=3,
        expanded_c=88,
        out_c=24,
        use_se=False,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=24,
        kernel=3,
        expanded_c=96,
        out_c=40,
        use_se=True,
        activation="HS",
        stride=2,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=40,
        kernel=3,
        expanded_c=240,
        out_c=40,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=40,
        kernel=3,
        expanded_c=240,
        out_c=80,
        use_se=True,
        activation="HS",
        stride=2,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=80,
        kernel=3,
        expanded_c=384,
        out_c=80,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=80,
        kernel=3,
        expanded_c=384,
        out_c=96,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=96,
        kernel=3,
        expanded_c=576,
        out_c=96,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=96,
        kernel=3,
        expanded_c=576,
        out_c=160,
        use_se=True,
        activation="HS",
        stride=2,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=160,
        kernel=3,
        expanded_c=960,
        out_c=160,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    ),
    InvertedResidualConfig(
        input_c=160,
        kernel=3,
        expanded_c=960,
        out_c=320,
        use_se=True,
        activation="HS",
        stride=1,
        width_multi=1.0
    )
]

model = MobileNetV3(
    inverted_residual_setting=inverted_residual_setting,
    last_channel=1280,
    num_classes=N_CLASSES,
    norm_layer=nn.BatchNorm2d
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

# 将模型和数据移到设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_loader = train_loader
test_loader = test_loader


# 定义训练函数
def train(model, loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in loader:
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

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)

    print(f'Epoch {epoch + 1}/{EPOCH}, Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc


# 定义评估函数
def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)

    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc


# 定义训练和评估循环
best_acc = 0.0

for epoch in range(EPOCH):
    train_loss, train_acc = train(model, train_loader, optimizer, epoch)
    val_loss, val_acc = evaluate(model, test_loader)

    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved.')

print(f'Best validation accuracy: {best_acc:.4f}')

# 最终测试
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')