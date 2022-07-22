---
yout:     post
title:      "关于职业发展的一些想法（2018.5.27汇总）"
subtitle:   ""
date:       2018-05-27 22:00:00
author:     "Hebi"
header-img: "img/ncu_sunset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - 职业发展
    - 所思所想
---

下面1-3是3月份找工作的时候写的，那时面试了5家企业，都失败了，内心很失落，也知道了自己的不足.

## 1. 关于工作的各因素排位

- 1). 是否是自己想做的（岗位预期）
- 2). 上班路程
- 3). 薪资（五险一金）
- 4). 直属领导在本岗位上的指导能力
- 5). 岗位发展前景（行业、项目落地场景）
- 6). 公司氛围

## 2. 你的职业规划是什么?

- 短期： 在目标检测上有所经验
- 中期：在图像（深度学习）
- 长期：成为技术大拿？


## 3. 你为什么想换工作？

- 1). 工作得不到突破，误检情况一直存在。现在的情况是，a). 通过加入patched样本后，误检大量减少。但是，对于施工或者外面进来的物体容易被误检为人，比如背包、圆形线圈. b). 安全帽的召回率还是很低，这跟resize的size有关.
c). 很长的一段时间，自己确立的短期目标是宁可漏检，不要误检，总之先解决误检问题. 因为在识别结果中很容易发现误检情况

> 关于人的误检，这是不是“hard begative”的问题. 现在解决误检的手段是，增加正样本，并增加制作好的误检样本(图片粘贴了hat-man块)，从而减少误检.

> 人的误检，当然还有其他途径可以解决

- 2). 上班路程远

- 3). 岗位无发展前景. 我不知道当时我为什么这么认为？可能是，没有那种做深度学习的氛围，没有这种环境，身边没有大拿，无法快速成长.

- 4). 直属领导无法有效地指导

- 5). 大boss不关注基本的东西——准确率. 现在关注了，而且提出要“零误检、零漏检”. 



## 4. 我的定位是什么？

2018.05.27. 18:12

对于深度学习，我的个人定位是什么？**专注应用（具体为图像识别与目标识别），并理解掌握算法.**

- 专注于图像方面，具体是图像识别与目标检测

- 专注于应用层面，致力于如何提高precision与recall

- 不求研究出怎样的算法，但是要研究算法，去了解算法背后运行的逻辑. **途径就是阅读文献，综合归纳与对比，形成自己的认知.**

- 在阅读文献的同时，思考——能否被应用到现有工作中，能否解决现有工作中的问题.

- 掌握深度学习的基本理论知识，不管光要理解掌握，还要动手写它们的naive code. 途径是**系统阅读深度学习方面的书籍，动手时间相关技术的naive code**.

- 掌握一两种深度学习框架，caffe，tensorflow，keras

- 具备模型训练的能力，包括调参，有报道说，调参的工作在以后会消失

- 在闲暇

## 4. 目前我需要做的事情（2018.5.27）

- 1). 阅读相关文献，总结归纳，形成自己的认知. 我已经好久没有读文献了

- 2). 阅读《深度学习》，掌握理论，实现其naive code.