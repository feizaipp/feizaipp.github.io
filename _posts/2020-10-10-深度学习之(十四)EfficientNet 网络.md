---
layout:     post
title:      深度学习之(十四)EfficientNet 网络
#subtitle:  
date:       2020-10-10
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - DeepLeaning
    - AI
    - Object Detective
---

> [我的博客](http://feizaipp.github.io)

# 1. 概述
&#160; &#160; &#160; &#160;EfficientNet 是谷歌在 2019 年提出的新的特征提取网络。它的主要创新点并不是结构，不像 ResNet 、 SENet 发明了 shortcut 或 attention 机制， EfficientNet 的 base 结构是利用结构搜索搜出来的，然后使用 compound scaling 规则放缩，得到一系列表现优异的网络： B0~B7 。

&#160; &#160; &#160; &#160;增加网络参数可以获得更好的精度（有足够的数据，不过拟合的条件下），例如 ResNet 可以加深从 ResNet-18 到 ResNet-200 。增加网络参数的方式有三种：深度、宽度和分辨率。深度是指网络的层数，宽度指网络中卷积的 channel 数，分辨率是指通过网络输入大小（例如从 112x112 到 224x224 ）。直观上来讲，这三种缩放方式并不独立。对于分辨率高的图像，应该用更深的网络，因为需要更大的感受野，同时也应该增加网络宽度来获得更细粒度的特征。之前增加网络参数都是单独放大这三种方式中的一种，并没有同时调整，也没有调整方式的研究。 EfficientNet 使用了 compound scaling 方法，统一缩放网络深度、宽度和分辨率。

# 2. 网络结构
&#160; &#160; &#160; &#160;EfficientNet B0~B7 网络结构由三个部分组成，分别是 Stem 、 MBConvBlock 和 Final Layers 。其中 Stem 就是标准的卷积、 BN 、 激活函数。 MBConvBlock 结构类似 MobileNetV3 的网络结构。 Final Layers 只是在 1x1 的卷积加上预测器。

# 3. 网络实现
&#160; &#160; &#160; &#160;首先，我们先看一下网络的超参数。

* params 前三个参数定义了 EfficientNet B0~B7 网络结构在三个维度的缩放比例，最后一个参数是 dropout_rate 。
```
params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}
```

* settings 变量定一个 MBConv 网络的结构，其中 t 表示输入通道的扩张系数；
* c 表示输出通道数；
* n 表示该模块的重复次数；
* s 表示 stride ；
* k 表示卷积核大小；
```
settings = [
        # t,  c, n, s, k
        [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
        [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
        [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
        [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
        [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
        [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
        [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
    ]
```

## 3.1. Stem 结构实现
&#160; &#160; &#160; &#160;Stem 就是标准卷积 、 BN 、激活函数结构。激活函数用的是 swish 。
```
class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]
```

## 3.2. MBConvBlock 结构实现
&#160; &#160; &#160; &#160;MBConvBlock 结构实现如下：
* self.use_residual: 在输入和输出相等，且 stride==1 的时候，增加残差网络
* 使用 1x1 卷积升维
* 进行深度可分离卷积
* 增加注意力机制
* 使用 1x1 卷积降维到输出维度
* 最后增加 BN 层
* 前向传播时，如果使用 use_residual ，则要使用 _drop_connect
```
class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)
```

## 3.3. Final Layers 结构实现
&#160; &#160; &#160; &#160;Final Layers 结构是 1x1 卷积和全链接层输出为类别的个数。

## 3.4. EfficientNet 网络的整体实现
&#160; &#160; &#160; &#160;初始化一个 EfficientNet 网络，传入参数为网络名称，例如： efficientnet_b0 。
```
def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)

def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model
```

&#160; &#160; &#160; &#160;EfficientNet 网络中根据不同的网络名称对网络的宽度和深度进行缩放。
* 宽度缩放： _round_filters ，缩放后的数字要能被 8 整除
* 深度缩放： _round_repeats ，缩放后的数字向上取整
* 
```
class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels

        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(in_channels, last_channels, 1)]

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
```