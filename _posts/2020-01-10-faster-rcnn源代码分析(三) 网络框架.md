---
layout:     post
title:      faster-rcnn源代码分析(三) 网络框架
#subtitle:  
date:       2020-01-10
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
&#160; &#160; &#160; &#160;这篇文章我们介绍一下 Faster-RCNN 网络的整体框架，以及 pytorch 的实现。 Faster-RCNN 网络主要由两部分组成， RPN 网络和 ROI 网络，下面分别介绍这两部分。

# 2. Faster-RCNN 网络介绍
&#160; &#160; &#160; &#160;创建 FasterRCNN 类，该类继承自 FasterRCNNBase 。
```
model = FasterRCNN(backbone=backbone, num_classes=91)
```

&#160; &#160; &#160; &#160;FasterRCNN 类的 init 函数中，主要创建了 RPN 网络和 ROI 网络，具体创建过程请看下一章节。

&#160; &#160; &#160; &#160;forword 函数在子类中实现。该函数执行整个网络的各个子模块。
* self.transform: 对输入的批量图像进行预处理
* self.backbone: 将图像输入到主干网络进行特征提取
* self.rpn: 将特征层以及标注 target 信息传入 rpn 中
* self.roi_heads: 将 rpn 生成的数据以及标注 target 信息传入 ROI 网络中
* self.transform.postprocess: 对网络的输出结果进行后处理，主要将 bboxes 还原到原图像尺度上
```
def forward(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")

    if self.training:
        assert targets is not None
        for target in targets:         # 进一步判断传入的target的boxes参数是否符合规定
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError("Expected target boxes to be a tensor"
                                        "of shape [N, 4], got {:}.".format(
                                        boxes.shape))
            else:
                raise ValueError("Expected target boxes to be of type "
                                    "Tensor, got {:}.".format(type(boxes)))

    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2  # 防止输入的是个一维向量
        original_image_sizes.append((val[0], val[1]))
    # original_image_sizes = [img.shape[-2:] for img in images]

    images, targets = self.transform(images, targets)  # 对图像进行预处理
    features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
    if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
        features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

    # 将特征层以及标注target信息传入rpn中
    proposals, proposal_losses = self.rpn(images, features, targets)

    # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

    # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)
```

## 2.1. RPN 网络
&#160; &#160; &#160; &#160;RPN 网络主要用于生成候选框，将生成的候选框输入到后续 ROI 网络中用于更精准的预测。RPN 由两部分组成，分别是 anchor 生成器和 RPN 预测网络。

&#160; &#160; &#160; &#160;anchor 生成器实现代码如下所示，每个预测特征层生成三个 anchor 。
```
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)
```

&#160; &#160; &#160; &#160;RPN 预测网络实现代码如下所示，预测网络中主要包括类别预测(前景或背景)，以及边界框回归参数预测。
```
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
```

&#160; &#160; &#160; &#160;整个 RPN 网络的实现由 RegionProposalNetwork 实现，代码如下：
```
rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
```

&#160; &#160; &#160; &#160;RPN 网络在生成候选框的过程中需要一些超参数，超参数定义及含义如下：
* rpn_fg_iou_thresh: 采集正样本设置的阈值
* rpn_bg_iou_thresh: 采集负样本设置的阈值
* rpn_batch_size_per_image: 采样的样本数
* rpn_positive_fraction: 正样本占总样本的比例
* rpn_pre_nms_top_n: rpn 中在 nms 处理前保留的 proposal 个数，这是每个预测特征层保留 proposal 的个数
* rpn_post_nms_top_n: rpn 中在 nms 处理后保留的 proposal 个数，这是每张图片总共保留的 proposal 的个数
* rpn_nms_thresh: rpn 中进行 nms 处理时使用的iou阈值

## 2.2. ROI 网络
&#160; &#160; &#160; &#160;ROI 网络有三部分组成，分别是ROI pooling 层、全连接层和预测层。

&#160; &#160; &#160; &#160;ROI pooling 层实现如下所示：
```
box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)
```

&#160; &#160; &#160; &#160;全连接层包括两个全连接层，实现如下：
```
resolution = box_roi_pool.output_size[0]  # 默认等于7
representation_size = 1024
box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
```

&#160; &#160; &#160; &#160;预测层包括两个全连接层，分别预测类别(真正的类别和背景)和边界框回归参数，实现如下：
```
representation_size = 1024
box_predictor = FastRCNNPredictor(representation_size, num_classes)
```
&#160; &#160; &#160; &#160;整个 ROI 网络的实现在 RoIHeads 中实现，具体代码如下所示：
```
roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
```

&#160; &#160; &#160; &#160;ROI 网络包括一些超参数，定义及含义如下：
* box_score_thresh: 移除低目标概率
* box_nms_thresh: fast rcnn 中进行nms处理的阈值
* box_detections_per_img: 对预测结果根据 score 排序取前 100 个目标
* box_fg_iou_thresh: fast rcnn 计算误差时，采集正样本设置的阈值
* box_bg_iou_thresh: fast rcnn 计算误差时，采集负样本设置的阈值
* box_batch_size_per_image: fast rcnn 计算误差时采样的样本数
* box_positive_fraction: fast rcnn 计算误差时采样正样本占所有样本的比例
* bbox_reg_weights: 计算误差时，边界框回归参数所占的权重

# 3. 数据预处理
&#160; &#160; &#160; &#160;在将一个批量图像数据送入主干网络之前要对批量图像进行预处理；在 ROI 网络输出的预测信息映射回原始图像上时也需要对图像进行处理，这些工作都是在数据预处理模块中完成的。

&#160; &#160; &#160; &#160;数据预处理类是 GeneralizedRCNNTransform 。

&#160; &#160; &#160; &#160;init 函数中，主要对图片预处理参数进行初始化。
* self.min_size=800: 指定图像的最小边长范围
* self.max_size=1344: 指定图像的最大边长范围
* self.image_mean = [0.485, 0.456, 0.406]: 指定图像在标准化处理中的均值
* self.image_std = [0.229, 0.224, 0.225]: 指定图像在标准化处理中的方差
```
def __init__(self, min_size, max_size, image_mean, image_std):
    super(GeneralizedRCNNTransform, self).__init__()
    if not isinstance(min_size, (list, tuple)):
        min_size = (min_size,)
    self.min_size = min_size      # 指定图像的最小边长范围
    self.max_size = max_size      # 指定图像的最大边长范围
    self.image_mean = image_mean  # 指定图像在标准化处理中的均值
    self.image_std = image_std    # 指定图像在标准化处理中的方差
```

&#160; &#160; &#160; &#160;forward 函数，对图像进行预处理操作。
* 遍历批量数据中的每一张图片，首先进行标准化: self.normalize
* 遍历批量数据中的每一张图片，对图片和图片中的目标进行 resize
* 记录 resize 后的图像的尺寸
* self.batch_images: 将 resize 后的图像打包成一个 batch
* image_sizes_list: 保存每一张 resize 后的图像的尺寸
* ImageList: 该类用于存储 resize 后的图像的 tensor 和图像的尺寸
* 函数返回 ImageList 对象和 resize 后的 targets
```
def forward(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
    images = [img for img in images]
    for i in range(len(images)):
        image = images[i]
        target_index = targets[i] if targets is not None else None

        if image.dim() != 3:
            raise ValueError("images is expected to be a list of 3d tensors "
                                "of shape [C, H, W], got {}".format(image.shape))
        image = self.normalize(image)                # 对图像进行标准化处理
        image, target_index = self.resize(image, target_index)   # 对图像和对应的bboxes缩放到指定范围
        images[i] = image
        if targets is not None and target_index is not None:
            targets[i] = target_index

    # 记录resize后的图像尺寸
    image_sizes = [img.shape[-2:] for img in images]
    images = self.batch_images(images)  # 将images打包成一个batch
    image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

    for image_size in image_sizes:
        assert len(image_size) == 2
        image_sizes_list.append((image_size[0], image_size[1]))

    image_list = ImageList(images, image_sizes_list)
    return image_list, targets
```

&#160; &#160; &#160; &#160;接下来介绍 normalize 函数。
* 首先将 self.image_mean 和 self.image_std 转化成 tensor 类型
* 将图片的每一个像素减去均值后再除以标准差

&#160; &#160; &#160; &#160;这里用到了矩阵运算的广播机制， mxn 的矩阵，加减乘除一个 1xn 的矩阵，通过复制 m 行来自动补全成 mxn 的矩阵，再进行运算；同理， mxn 的矩阵，加减乘除一个 mx1 的矩阵，通过复制 n 列来自动补全成 mxn 的矩阵，再进行计算。这里我们图片是 3xmxn 维的矩阵，标准化时要将 mean 和 std 进行升维，将 shape 为 [3] 的 tensor 变为 [3, 1, 1] 的 tensor 。
```
def normalize(self, image):
    """标准化处理"""
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
    # [:, None, None]: shape [3] -> [3, 1, 1]
    return (image - mean[:, None, None]) / std[:, None, None]
```

&#160; &#160; &#160; &#160;self.resize 函数用来对一个批量的图像进行 resize 到相同的大小，并对 target 进行等比例的缩放。
* 将指定缩放的最小边长除以图像的最小边长，得到缩放因子
* 用图像的最长边乘以上述缩放因子，如果得到的值大于指定的最大边长，则缩放因子等于指定的最大边长除以图像的最大边长
* 利用插值的方法缩放图片，这里注意 bilinear 方法只支持 4D 的 tensor ，所用通过 image[None] 操作将图像转化为 4D 。
* 使用 resize_boxes 函数对 target 进行缩放。
```
def resize(self, image, target):
    # type: (Tensor, Optional[Dict[str, Tensor]])

    # image shape is [channel, height, width]
    h, w = image.shape[-2:]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))  # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))  # 获取高宽中的最大值
    if self.training:
        size = float(self.torch_choice(self.min_size))  # 指定输入图片的最小边长,注意是self.min_size不是min_size
    else:
        # FIXME assume for now that testing uses the largest scale
        size = float(self.min_size[-1])    # 指定输入图片的最小边长,注意是self.min_size不是min_size
    scale_factor = size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self.max_size:
        scale_factor = self.max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

    if target is None:
        return image, target

    bbox = target["boxes"]
    # 根据图像的缩放比例来缩放bbox
    bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
    target["boxes"] = bbox

    return image, target
```

&#160; &#160; &#160; &#160;resize_boxes 对 target 进行缩放。
* 输入参数分别分要缩放的 target 、原始图像的大小、缩放后的图像的大小
* ratios: 记录宽高的缩放比例，用缩放后的图片的除以原始图片
* boxes.unbind(1): 移除维度 1 ，返回一个元组，包含了沿着指定维切片后的各个切片。
* 缩放对应 target 的边界框参数
* 将缩放后的 target 边界框参数堆叠到一起
```
def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor

    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
```

&#160; &#160; &#160; &#160;下面介绍 batch_images 函数，该函数是把一个批量大小的图像数据进行打包。因为每一个批量中各个图像的大小是不相同的，该函数是找出一个批量数据中所有维度的最大值作为批量数据的维度。将这些维度的最大值 32 字节向上对齐，然后生成一个同样维度的值为 0 的批量矩阵。然后将每一张图像的值复制到新生成的批量矩阵中。
* self.max_by_axis: 分别计算一个 batch 中所有图片中的最大 channel height width
* size_divisible: 向上取整的步数，取整为了对硬件友好
* new_full(batch_shape, 0): 生成一个 batch_shape 形状的 tensor ，设置默认值为 0
* 遍历每一张图像，将像素值写入新生成的 tensor 中
```
def batch_images(self, images, size_divisible=32):
    # type: (List[Tensor], int)

    if torchvision._is_tracing():
        # batch_images() does not export well to ONNX
        # call _onnx_batch_images() instead
        return self._onnx_batch_images(images, size_divisible)

    # 分别计算一个batch中所有图片中的最大channel, height, width
    max_size = self.max_by_axis([list(img.shape) for img in images])

    stride = float(size_divisible)
    # max_size = list(max_size)
    # 将height向上调整到stride的整数倍
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    # 将width向上调整到stride的整数倍
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    # [batch, channel, height, width]
    batch_shape = [len(images)] + max_size

    # 创建shape为batch_shape且值全部为0的tensor
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
        # 这样保证输入到网络中一个batch的每张图片的shape相同
        # copy_: Copies the elements from src into self tensor and returns self
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs
```
