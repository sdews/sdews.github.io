---
layout:      post
title:       "目标检测文献阅读与理解之Faster R-CNN"
subtitle:    ""
date:        2018-07-27 16:09
author:      "HeBi"
header-img:  "img/ncu_sunset.jpg"
header-mask: 0.3
catalog:     true
tags:
    - 深度学习
    - 图像识别
    - 目标检测
---

论文题目: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)


论文作者： Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun

提交时间： 2015-06-04(v1), 2016-01-06(v3, last)
	
## Faster R-CNN

### Faster R-CNN (test time)

Faster R-CNN (test time) 由三部分组成：

- 特征提取网络 ConvNet. 输入是 Image, 输出是 Conv feature maps

> ConvNet 的基础网络, 作者选用了三种网络(论文里用了两种, 源码里用了三种):
>
> > the Zeiler and Fergus model: 简称 ZF, 使用了`5`个卷积层
> >
> > > [input-data] - [conv/relu - norm - pool] * 2 - [conv/relu] * 3
> >
> > the Simonyan and Zisserman model: 简称 VGG-16, 使用了`13`个卷积层
> >
> > > [conv/relu - conv/relu - pool] * 2 - [conv/relu - conv/relu - conv/relu - pool] * 2 - [conv/relu - conv/relu - conv/relu]
> > 
> > VGG-CNN-M-1024: 使用了`5`层卷积层
> >
> > > [input-data] - [conv/relu - norm - pool] * 2 - [conv/relu] * 3

- Region Proposal Network. 输入是 Conv feature map, 输出是 Proposals

- 分类与定位, 包含ROIPooling, fc, softmax layer. 输入是conv feature map, Proposals, 输出是 class scores, bbox predictions

Faster R-CNN (test time) 的流程图如下：

![image](/img/in-post/faster_r-cnn/faster_rcnn-architecture.png)

注意到, Fast R-CNN (test time) 的流程图如下:

![image](/img/in-post/fast-rcnn/fast_r-cnn_test-time.png)

> 对比后可见, 相比于 Fast R-CNN (test time), Faster R-CNN (test time) 用 RPN 取代了 额外的 Proposal generating method, 将 RPN 直接接入了网络之中. 

### Faster R-CNN (train time)




### 运行阶段的网络结构

## Region Proposal Network

Region Proposal Network, 下面简称为RPN, 这是较Fast R-CNN增加的部分, 其取代了Selective search方法

### RPN的任务——用来做什么的？


### RPN的结构

#### 训练阶段

#### 运行阶段

#### anchor box


### RPN的训练


#### 定义正负样本及无效样本


#### 取样规则


#### $Loss$函数


