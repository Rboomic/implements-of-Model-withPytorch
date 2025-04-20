import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 将我们传入的channel的个数转换为距离8最近的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 卷积+BN+激活函数模块
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size: int = 3,  # 卷积核大小
                 stride: int = 1,
                 groups: int = 1,  # 用来控制我们深度可分离卷积的分组数(DWConv：这里要保证输入和输出的channel不会发生变化)
                 norm_layer: Optional[Callable[..., nn.Module]] = None,  # BN结构
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        # super() 函数接受两个参数：子类名和子类对象，用来指明在哪个子类中调用父类的方法。在这段代码中，ConvBNActivation 是子类名，self 是子类对象。
        # 通过super(ConvBNActivation, self)，Python知道要调用的是ConvBNActivation的父类的方法。
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=input_channel,
                                                         out_channels=output_channel,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(output_channel),
                                               activation_layer())


# SE模块：注意力机制
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_channel: int,  # block input channel
                 expand_channel: int,  # block expand channel 第一个1X1卷积扩展之后的channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_channel // squeeze_factor  # 第一个全连接层个数等于输入特征的1/4
        self.fc1 = nn.Conv2d(expand_channel, squeeze_c, 1)  # 压缩特征
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_channel, 1)  # 拓展特征
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 全局平均池化
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x  # 与输入的特征进行相乘


# 每个MBconv的配置参数
class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5 论文中的卷积核大小有3和5
                 input_channel: int,
                 out_channel: int,
                 expanded_ratio: int,  # 1 or 6 #第一个1x1卷积层的扩展倍数，论文中有1和6
                 stride: int,  # 1 or 2
                 use_se: bool,  # True 因为每个MBConv都使用SE模块 所以传入的参数是true
                 drop_rate: float,  # 随机失活比例
                 index: str,  # 1a, 2a, 2b, ... 用了记录当前MBconv当前的名称
                 width_coefficient: float):  # 网络宽度的倍率因子
        self.input_c = self.adjust_channels(input_channel, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_channel, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    # 后续如果想要继续使用B1~B7，可以使用B0的channel乘以倍率因子
    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


# 搭建MBconv模块
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)  # 当满足两个条件之后才能使用shortcut连接

        layers = OrderedDict()  # 创建空的有序字典用来保存MBConv
        activation_layer = nn.SiLU  # alias Swish

        # 搭建1x1升维卷积 ：这里其实是有个小技巧，论文中的MBconv的第一个1x1卷积是为了做升维操作，如果我们的expand为1的时候，可以不搭建第一个卷积层
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise 搭建深度可分离卷积（这里要保证输入和输出的channel不会发生变化）
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,  # 只有保证分组数和输入通道数保持一致才能确保输入和输入的channel保持不变
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})  # Identity 不做任何激活处理

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 9,
                 dropout_rate: float = 0.2,  # 网络中最后一个全连接层的失活比例
                 drop_connect_rate: float = 0.2,  # 是MBconv中的随机失活率
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(input_channel=1,
                                                     output_channel=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(input_channel=last_conv_input_c,
                                               output_channel=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
model = EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, num_classes=9)

# 确保使用相同的随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义设备（GPU优先）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据预处理和加载部分保持不变

# 修改后的数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

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
        dataset = torch.utils.data.TensorDataset(features, labels)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

# 创建数据加载器
train_loader = MyDataset(X_train_tensor, y_train_tensor, BATCH_SIZE, shuffle=True)
test_loader = MyDataset(X_test_tensor, y_test_tensor, BATCH_SIZE, shuffle=False)

# 初始化模型
model = EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, num_classes=9).to(device)

# 定义训练工具
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

# 训练循环
best_accuracy = 0.0
for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{EPOCH}] Loss: {epoch_loss:.4f}')

    # 计算验证准确率
    model.eval()
    val_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_accuracy += torch.sum(preds == labels.data).item()
    val_accuracy = val_accuracy / len(test_loader.dataset)
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved.')

# 测试模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_accuracy = 0.0
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        test_accuracy += torch.sum(preds == labels.data).item()
test_loss = test_loss / len(test_loader.dataset)
test_accuracy = test_accuracy / len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 评估模型性能
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
#def efficientnet_b0(num_classes=1000):
#    # input image size 224x224
#    return EfficientNet(width_coefficient=1.0,
#                        depth_coefficient=1.0,
#                        dropout_rate=0.2,
#                        num_classes=num_classes)


#def efficientnet_b1(num_classes=1000):
    # input image size 240x240
#    return EfficientNet(width_coefficient=1.0,
#                        depth_coefficient=1.1,
#                        dropout_rate=0.2,
#                        num_classes=num_classes)


#def efficientnet_b2(num_classes=1000):
    # input image size 260x260
#    return EfficientNet(width_coefficient=1.1,
#                        depth_coefficient=1.2,
 #                       dropout_rate=0.3,
 #                       num_classes=num_classes)


#def efficientnet_b3(num_classes=1000):
    # input image size 300x300
#    return EfficientNet(width_coefficient=1.2,
#                        depth_coefficient=1.4,
#                        dropout_rate=0.3,
 #                       num_classes=num_classes)


#def efficientnet_b4(num_classes=1000):
    # input image size 380x380
#    return EfficientNet(width_coefficient=1.4,
#                        depth_coefficient=1.8,
#                        dropout_rate=0.4,
#                       num_classes=num_classes)


#def efficientnet_b5(num_classes=1000):
    # input image size 456x456
#    return EfficientNet(width_coefficient=1.6,
#                        depth_coefficient=2.2,
#                        dropout_rate=0.4,
#                       num_classes=num_classes)


#def efficientnet_b6(num_classes=1000):
    # input image size 528x528
#    return EfficientNet(width_coefficient=1.8,
#                        depth_coefficient=2.6,
#                        dropout_rate=0.5,
#                        num_classes=num_classes)


#def efficientnet_b7(num_classes=1000):
#    # input image size 600x600
#    return EfficientNet(width_coefficient=2.0,
#                        depth_coefficient=3.1,
#                        dropout_rate=0.5,
#                        num_classes=num_classes)

