---
layout:     post
title:      faster-rcnn源代码分析(二)
#subtitle:  
date:       2019-12-28
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
&#160; &#160; &#160; &#160;上一篇介绍了 faster-rcnn 数据获取流程，本篇文章介绍 anchor 生成过程。 faster-rcnn 提出了 anchor 思想用来生成候选区域， anchor 生成在训练过程和测试过程都要用到。

&#160; &#160; &#160; &#160;faster-rcnn 的 caffe 实现中，生成 anchor 是用单独的一个 Layer 来完成的，该层就是 AnchorTargetLayer 。在 caffe 运行模型的过程中，首先执行层的 setup 函数。
```
#py-faster-rcnn/lib/rpn/anchor_target_layer.py
def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8, 16, 32)) #............(1)
        self._anchors = generate_anchors(scales=np.array(anchor_scales)) #..........(2)
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:] #............(3)
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors #...................(4)
        # labels
        top[0].reshape(1, 1, A * height, width) #...........(5)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)
```
* 1.读取 prototxt 文件中的 scales 参数，如果不存在则默认值为 (8, 16, 32) ，该值表示 anchor 的尺寸。
* 2.根据尺寸生成 anchor。
* 3.根据输入图片的数据获得图片的宽和高，图片数据的最后两个维度存储的是图片的高和宽。
* 4.将生成的 anchor 数量赋值给 A
* 5.返回数据，供下一层使用，top[0] 表示 anchor 对应类别的，有两种，前景或背景； top[1] 表示 anchor 的 bbox ； top[2~3] 表示 bbox 的权重。

&#160; &#160; &#160; &#160;下面重点分析生成 anchor的流程。
```
#py-faster-rcnn/lib/rpn/anchor_target_layer.py
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1 #...........(1)
    ratio_anchors = _ratio_enum(base_anchor, ratios) #....................(2)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])]) #..............(3)
    return anchors
```
* 1.生成 base_anchor ，为 [0, 0, 16, 16]
* 2.基于 base_anchor 生成宽高比为 [0.5, 1, 2] 的 3 个 anchor
* 3.在第二步生成的三个 anchor 基础上再次生成尺度为 (8, 16, 32) 的 9 个 anchor

&#160; &#160; &#160; &#160;下面分析基于 base_anchor 的 anchor 生成流程。
```
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor) #..........(1)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios) #.....................(2)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr) #...........(3)
    return anchors
```
* 1.根据 anchor 的左上角坐标和右下角坐标，得到 anchor 的宽高和中心点坐标
* 2.根据 ratios 和 base_anchor 的宽高、中心点坐标，得到三种宽高的 anchor ，计算公式如上述代码所示，此时的 ws hs 为三种 anchor 的宽和高，中心点没变。这么计算三种宽高比的原因是为了保证最终生成的 3 个不同宽高比的 anchor 的面积尽量接近
* 3.第二步中得到的是 anchor 的宽高和中心点坐标，这一步是要转换回左上角和右下角坐标形式

&#160; &#160; &#160; &#160;下面是 _whctrs 的代码，理解不了的话就在纸上画一画就明白了。
```
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr
```
&#160; &#160; &#160; &#160;下面是 _mkanchors 的代码，理解不了的话就在纸上画一画就明白了。
```
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors
```
&#160; &#160; &#160; &#160;上面介绍了通过 base_anchor 生成 3 个 anchor 的坐标，对于多尺度的 anchor 坐标生成是通过 _scale_enum 函数实现的。该函数与 _ratio_enum 基本相同，只是扩展了三个尺度。
```
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
```
&#160; &#160; &#160; &#160;以上介绍完了基于 base_anchor 生成 3 种尺度， 3 种宽高比共 9 个 anchor ，下面介绍网络的前向传播，最终针对 featrue map 的每个点都生成 9 个 anchor ，并对 anchor 进行筛选。
```
def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights
```