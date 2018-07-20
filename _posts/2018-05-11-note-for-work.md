---
layout:     post
title:      "2018-05-11-工作笔记"
subtitle:   ""
date:       2018-05-11 22:45:00
author:     "Hebi"
header-img: "img/ncu_sunset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - 工作笔记
    - hard negatives
    - 目标检测
---


## 关于resize前后计算均值
实验显示，resize成同一规格（指定短边大小与长边最大值）后计算rgb均值，或者不resize直接计算rgb均值，结果在0.1尺度上相等

## 解决hard negatives的土办法
下一步也打算考虑anchor问题（即解决小目标漏检），目前着重解决hard negatives，解决办法是
- 法一，增加包含hard negatives且包含目标物的图片为训练样本
- 法二，把含hard negatives的图片取过来（同事帮我做了个功能，支持下载指定站点-摄像头-检测项目的检测原图），先准备好一批含目标物的patch image（patch image是从图片里截图得来的，比较小，刚好包含目标物），并准备好对应的patch xml，然后随机抽取patch image，把它粘贴到hard negatives图片的指定位置上，并生成新的xml，这样就制作好了训练样本。以上是批量处理。重新基于训练集进行训练，这样可以减小hard negatives的scores

2018-05-11

