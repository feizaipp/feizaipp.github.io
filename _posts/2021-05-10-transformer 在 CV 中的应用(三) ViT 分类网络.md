---
layout:     post
title:      transformer 在 CV 中的应用(三) ViT 分类网络
#subtitle:  
date:       2021-05-10
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - DeepLeaning
    - AI
    - Transformer
    - Object Classification
---

> [我的博客](http://feizaipp.github.io)

# 0. 参考资料
* [transformer 在 CV 中的应用(一) Transformer 介绍](https://feizaipp.github.io/2021/03/22/transformer-%E5%9C%A8-CV-%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8(%E4%B8%80)-Transformer-%E4%BB%8B%E7%BB%8D)

# 1. 概述
&#160; &#160; &#160; &#160;ViT 网络是谷歌在 2020 年提出的基于纯 Transformer 实现的分类网络，它完全抛弃了 CNN 网络。 ViT 网络中的 Transformer 与传统意义上的 Transformer 存在明显的不同，传统的 Transformer 是用于 NLP 中的机器翻译任务中，它的结构由 Encoder 和 Decoder 两部分组成，因为要将输入的序列通过 Encoder 网络进行编码，然后将编码后的序列通过 Decoder 解码，最终得到目标语言。但是在视觉任务中，我们只需要对图像进行特征提取，然后将特征通过全链接层输出目标类别。

&#160; &#160; &#160; &#160;我们知道，在 NLP 模型中 Transformer 的输入是一个序列，那么对于图像数据， ViT 是将一张图像分成一个个小的 patch ，然后对这些 patch 进行编码。

&#160; &#160; &#160; &#160;ViT 网络的特征提取是使用了一个叫做 class token 的结构，该结构是可学习的。它与图像编码后的张量在 dim=1 处进行 cat 操作得到一个新的张量，新张量的第一个元素就是 class token ，这个新的张量与位置编码进行相加后输入到 ViT 网络中，最后学习到的 class token 即为图像的特征图，将这个特征通过全链接层得到类别的输出。

# 2. 网络结构
&#160; &#160; &#160; &#160;