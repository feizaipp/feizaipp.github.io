---
layout:     post
title:      基于 sort 算法的多目标跟踪
#subtitle:  
date:       2021-03-09
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - DeepLeaning
    - AI
    - Multi Object Tracker
---

> [我的博客](http://feizaipp.github.io)

# 0. 参考资料
* [详解Transformer](https://zhuanlan.zhihu.com/p/48508221)
* [Vision Transformer 超详细解读 (原理分析+代码解读) (一)](https://zhuanlan.zhihu.com/p/340149804)

# 1. 概述
&#160; &#160; &#160; &#160;Transformer 是 Google 的团队在 2017 年提出的一种 NLP 经典模型，经过近几年的发展， Transformer 不仅在 NLP 领域有很好的应用，在 CV 领域也得到了快速发展。最初 Transformer 是为了替换 RNN 网络而出现的，因为 RNN 循环神经网络是顺序模型，时间 t 时刻的计算依赖于时间 t-1 时刻的输出，这限制了模型的并行能力；其次顺序计算的过程中信息会丢失。为了解决上述两个问题， Transformer 使用了 attension 机制，将序列中任意两个位置之间的距离缩小为一个常量。；其次它避免了使用顺序结构，因此具有更好的并行性。

