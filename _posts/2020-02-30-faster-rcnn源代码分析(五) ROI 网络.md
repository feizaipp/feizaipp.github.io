---
layout:     post
title:      faster-rcnn源代码分析(五) ROI 网络
#subtitle:  
date:       2020-02-30
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
&#160; &#160; &#160; &#160;本篇文章介绍 ROI 网络。经过 RPN 网络生成的候选框输入进 ROI 网络， ROI 网络进行类别的判断，并对边界框进一步回归。

&#160; &#160; &#160; &#160;ROI 网络分三个部分，分别为 ROI Align 、 全链接层和预测层。其中 ROI Align 是 ROI pooling 的改进版。

# 2. ROI 网络结构
## 2.1. ROI Align
&#160; &#160; &#160; &#160;在 Faster RCNN 原论文中使用的 ROI pooling 层对预测特征层进行处理，但 ROI pooling 层在计算的过程中有两次整数化的过程，首先是对 RPN 网络生成的候选框进行取整操作；其次是将整数化后的边界区域平均分割成 k*k 个单元，对每个单元的边界进行整数化。经过上述两次整数化，此时的候选框已经和最开始回归出来的位置有一定的偏差，最终导致预测的精度。

&#160; &#160; &#160; &#160;ROI Align 层取消了整数化的过程，使用双线性插值的方法计算坐标为浮点数时的像素值。

&#160; &#160; &#160; &#160;这里还有一个问题，对于原始的 Faster RCNN 网络，只在一个 feature map 上进行预测，对于使用了 FPN 的 backbone 网络，是在多个 featrue map 上进行预测的。那么 ROI Align 层首先需要先计算每一个候选框使用哪个特征层进行预测。我们首先看下代码实现。

&#160; &#160; &#160; &#160; ROI Align 实现在类 MultiScaleRoIAlign 中。
* self.featmap_names: 表示在哪些特征层进行 ROI Align
* self.sampling_ratio: 采样点，默认是 2 ，对于 7*7 的每一个区域平分 2 份，分一份取中心点位置，中心点坐标采用双线性插值法进行计算，最终取每一份的最大值作为这个区域的像素值
* self.output_size: 候选框分割的大小，默认是 7*7
* self.scales: 存储每个 feature_map 相对于网络输入 image 的下采样倍率 scale
* self.map_levels: 存储所有 box 对应的 feature_map
```
class MultiScaleRoIAlign(nn.Module):
    __annotations__ = {
        'scales': Optional[List[float]],
        'map_levels': Optional[LevelMapper]
    }

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
```

&#160; &#160; &#160; &#160;前向传播函数如下。
* x_filtered: 保存参与预测的预测特征层
* num_levels: 保存参与预测的预测特征层的个数，如果是 1 则直接在该预测特征层上进行预测
* self.convert_to_roi_format: 将一个批量的所有图像的候选框合并在一起。并在每一张图像的候选框前加以列，标示图像编号。
* self.setup_scales: 计算 self.scales ，每个 feature_map 相对于网络输入 image 的下采样倍率 scale ；
* 如果网络只使用一个预测特征层，则直接进行调用 roi_align 函数进行预测。
* levels: 存储每一个预选框对应的预测特征层
* num_rois: rois 的个数
* num_channels: 预测特征层的通道数
* result: 创建一个 shape 为 (num_rois, num_channels, 7, 7) 的 tensor ，并初始化为 0
* 遍历每一个预测特征层和 scale ， idx_in_level 表示在该层预测的预选框的索引， rois_per_level 表示在该层预测的预选框的坐标值
* roi_align: 进行 roi align 处理，返回值是当前预测特征层对每个预选框进行 roi align 后的值
* result: 将每个预测特征层进行 roi align 处理后的结果保存到 result 中
```
def forward(self, x, boxes, image_shapes):
    # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]])
    x_filtered = []
    for k, v in x.items():
        if k in self.featmap_names:
            x_filtered.append(v)
    num_levels = len(x_filtered)
    rois = self.convert_to_roi_format(boxes)
    if self.scales is None:
        self.setup_scales(x_filtered, image_shapes)

    scales = self.scales
    assert scales is not None

    if num_levels == 1:
        return roi_align(
            x_filtered[0], rois,
            output_size=self.output_size,
            spatial_scale=scales[0],
            sampling_ratio=self.sampling_ratio
        )

    mapper = self.map_levels
    assert mapper is not None

    levels = mapper(boxes)

    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]

    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    result = torch.zeros(
        (num_rois, num_channels,) + self.output_size,
        dtype=dtype,
        device=device,
    )

    tracing_results = []
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.nonzero(levels == level).squeeze(1)
        rois_per_level = rois[idx_in_level]

        result_idx_in_level = roi_align(
            per_level_feature, rois_per_level,
            output_size=self.output_size,
            spatial_scale=scale, sampling_ratio=self.sampling_ratio)

        if torchvision._is_tracing():
            tracing_results.append(result_idx_in_level.to(dtype))
        else:
            result[idx_in_level] = result_idx_in_level

    if torchvision._is_tracing():
        result = _onnx_merge_levels(levels, tracing_results)

    return result
```

&#160; &#160; &#160; &#160;接下来看一下 convert_to_roi_format 函数，该函数是将一个批量图像生成的预选框合并在一起，并在第 1 维度前加上图像的索引。
```
def convert_to_roi_format(self, boxes):
    # type: (List[Tensor])
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois
```

&#160; &#160; &#160; &#160;下面介绍生成下采样倍率 scale 的函数，主要是在 infer_scale 函数中。该函数在 setup_scales 中调用。
* 2 ** (feature_size/original_size).log2() = [1/4, 1/8, 1/16, 1/32] ，不是很明白为什么要这么算一下，不绕吗...
```
def infer_scale(self, feature, original_size):
    # type: (Tensor, List[int])
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales = torch.jit.annotate(List[float], [])
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    assert possible_scales[0] == possible_scales[1]
    return possible_scales[0]
```
&#160; &#160; &#160; &#160;setup_scales 函数除了生成下采样倍率外，还初始化了 self.map_levels 变量，改变两用来身成每个预选框在哪个特征层进行预测。
* scales: 生成下采样的倍率
* lvl_min: 根据下采样倍率的最小值生成一个 lvl_min 变量。
* lvl_max: 根据下采样倍率的最小值生成一个 lvl_max 变量。
* self.map_levels: 用 lvl_min 和 lvl_max 初始化 initLevelMapper ， lvl_min=2;lvl_max=4
```
def setup_scales(self, features, image_shapes):
    # type: (List[Tensor], List[Tuple[int, int]])
    assert len(image_shapes) != 0
    max_x = 0
    max_y = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
    original_input_shape = (max_x, max_y)

    scales = [self.infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
    self.scales = scales
    self.map_levels = initLevelMapper(int(lvl_min), int(lvl_max))
```
&#160; &#160; &#160; &#160;下面介绍下，如何将预选框映射到某一个预测特征层：
```
def initLevelMapper(k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
    # type: (int, int, int, int, float)
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)
```

&#160; &#160; &#160; &#160;LevelMapper 类的定义年如下：
* self.k_min: 根据下采样倍率的最小值生成一个常量
* self.k_max: 根据下采样倍率的最大值生成一个常量
* self.s0: 常量，默认是 224
* self.lvl0: 常量，默认是 4
* self.eps: 常量，防止除数为 0
```
class LevelMapper(object):
    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        # type: (int, int, int, int, float)
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps
```
&#160; &#160; &#160; &#160;LevelMapper 类的前向传播函数定义如下：
* 将预选框映射到某一个预测特征层公式如下：k = k0 + log2(sqrt(wh)/224)
* 将得到的 target_lvls 值，通过 clamp 函数控制在 self.k_min 和 self.k_max 之间
* 最终结果要用 target_lvls - self.k_min
```
def __call__(self, boxlists):
    # type: (List[Tensor])
    """
    Arguments:
        boxlists (list[BoxList])
    """
    # Compute level ids
    s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

    # Eqn.(1) in FPN paper
    target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
    target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
    return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)
```
&#160; &#160; &#160; &#160;至此， MultiScaleRoIAlign 部分就介绍到这里。

## 2.2. 全链接层
&#160; &#160; &#160; &#160;经过 ROI Align 层后，将处理结果送入两层全链接层。

&#160; &#160; &#160; &#160;第一层全链接层的输入是 num_channels*7*7 ，输出是 1024 ；第二层全链接层输入是 1024 ，输出也是 1024 。

&#160; &#160; &#160; &#160;在正向传播过程中，先将 rois 的信息在第一维度上进行展平操作（因为每一个 rois 都要进行预测和边界框回归），然后进行两次全链接层。最终输出 1024 维的特征向量。
```
class TwoMLPHead(nn.Module):

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x
```

## 2.3. 预测层
&#160; &#160; &#160; &#160;网络经过两层全链接层的处理后，输出 1024 维的特征向量，将该特征向量输入预测层进行类别的预测和边界框的回归。

&#160; &#160; &#160; &#160;类别预测输出是 num_classes ，边界框回归预测输出是 num_classes * 4 。
```
class FastRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
```

# 3. ROI 网络实现
&#160; &#160; &#160; &#160;网络实现如下所示：

&#160; &#160; &#160; &#160;初始化 ROI Align 网络：
```
box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
            output_size=[7, 7],
            sampling_ratio=2)
```
&#160; &#160; &#160; &#160;初始化全链接网络：
```
resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )
```
&#160; &#160; &#160; &#160;初始化预测网络：
```
representation_size = 1024
box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)
```
&#160; &#160; &#160; &#160;创建整个 ROI 网络：
```
roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
```

&#160; &#160; &#160; &#160;先看 RoIHeads 定义。
* box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本
* fg_iou_thresh 和 bg_iou_thresh: 表示采集正负样本设置的阈值
* batch_size_per_image 和 positive_fraction: 表示采样的样本数，以及正样本占所有样本的比例
* score_thresh: 表示要移除低目标概率的值
* nms_thresh: 表示进行 nms 处理的阈值
* detection_per_img: 表示对预测结果根据 score 排序取前 100 个目标
```
class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detection_per_img):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 0.5
            bg_iou_thresh,  # 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # 512
            positive_fraction)     # 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img
```
&#160; &#160; &#160; &#160;正向传播函数如下：
* 判断 targets 类型是否正确，如果是训练模式，使用 self.select_training_samples 划分正负样本
* 调用 self.box_roi_pool 进行 ROI Align
* 调用 self.box_head 进行全链接层
* 调用 self.box_predictor 进行预测
* fastrcnn_loss: 如果是训练模式，计算损失
* self.postprocess_detections: 如果是预测模式，将预测值映射到真实图像得到预测类别以及边界框的位置
```
def forward(self, features, proposals, image_shapes, targets=None):
    # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])

    # 检查targets的数据类型是否正确
    if targets is not None:
        for t in targets:
            floating_point_types = (torch.float, torch.double, torch.half)
            assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
            assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

    if self.training:
        # 划分正负样本，统计对应gt的标签以及边界框回归信息
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    # 将采集样本通过roi_pooling层
    box_features = self.box_roi_pool(features, proposals, image_shapes)
    # 通过roi_pooling后的两层全连接层
    box_features = self.box_head(box_features)
    # 接着分别预测目标类别和边界框回归参数
    class_logits, box_regression = self.box_predictor(box_features)

    result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
    losses = {}
    if self.training:
        assert labels is not None and regression_targets is not None
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets)
        losses = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg
        }
    else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

    return result, losses
```
&#160; &#160; &#160; &#160;下面看一下正负样本划分函数 self.select_training_samples ，这里的正负样本的划分与 rpn 网络没有本质的区别，只是一些参数上的调整。
* self.add_gt_proposals: 将 gt boxes 与 proposals 预选框进行拼接
* self.assign_targets_to_proposals: 遍历每张图像，为每个 proposal 匹配对应的 gt box ，并划分到正负样本中
* self.subsample: 按给定数量和比例采样正负样本
* 遍历每一张图片，计算正负样本与真实标签的类别损失和边界框回归损失
* 返回处理后 proposals ， 正负样本对应的索引 matched_idxs ， 正负样本的预测类别 labels ， 以及正负样本的边界框回归参数 regression_targets
```
def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

        # 检查target数据是否为空
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的预测类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets
```
&#160; &#160; &#160; &#160;继续看孙是计算函数 fastrcnn_loss 。
* 类别损失使用多分类的交叉熵损失函数
* 边界框回归参数使用 smooth_l1 损失
```
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框损失信息
    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss
```
&#160; &#160; &#160; &#160;如果是预测模式，调用 postprocess_detections 对预测的类别和边界框进行处理。
* self.box_coder.decode: 根据 proposal 以及预测的回归参数计算出最终 bbox 坐标
* F.softmax: 对预测类别结果进行 softmax 处理
* box_ops.clip_boxes_to_image: 裁剪预测的 boxes 信息，将越界的坐标调整到图片边界上
* boxes = boxes[:, 1:],scores = scores[:, 1:],labels = labels[:, 1:]: 移除所有背景信息
* torch.nonzero(scores > self.score_thresh).squeeze(1): 移除低概率目标
* box_ops.remove_small_boxes: 移除小目标
* box_ops.batched_nms: 执行 nms 处理，并按 scores 进行排序
* keep[:self.detection_per_img]: 根据 scores 排序返回前 topk 个目标
```
def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
    # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])

    device = class_logits.device
    # 预测目标类别数
    num_classes = class_logits.shape[-1]

    # 获取每张图像的预测bbox数量
    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    # 根据proposal以及预测的回归参数计算出最终bbox坐标
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    # 对预测类别结果进行softmax处理
    pred_scores = F.softmax(class_logits, -1)

    # split boxes and scores per image
    # 根据每张图像的预测bbox数量分割结果
    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    # 遍历每张图像预测信息
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove prediction with the background label
        # 移除索引为0的所有信息（0代表背景）
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        # 移除小目标
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximun suppression, independently done per class
        # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

        # keep only topk scoring predictions
        # 获取scores排在前topk个预测目标
        keep = keep[:self.detection_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels
```

&#160; &#160; &#160; &#160;至此， ROI 网络就介绍完了。