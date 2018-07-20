---
layout:     post
title:      "[已解决]如何在kramdown中插入数学公式并在这个博客中展现出来"
subtitle:   ""
date:       2018-07-15 18:12:00
author:     "Hebi"
header-img: "img/ncu_sunset.jpg"
header-mask: 0.3
catalog:    true
tags:
    - kramdown
    - 数学公式
---

## 公式无法显示

之前的笔记是在有道云笔记用markdown写的，它使用了katex, 把那里的笔记拷过来，公式在这里均无法显示，咋办？

是不是要在js里写啥东西？前端的东西一点都不懂。

算了，我还是继续用有道吧，就是无法在博客中展现出来，以后再找找看，有没有支持数学公式的现成模板


写于2018-07-16

* * *
---

## 问题已解决

注意到，我使用的是来自Hux的网页模板，其也是源自Jekyll。

后面，我耐心地Google了[相关问题](https://stackoverflow.com/questions/26275645/how-to-supported-latex-in-github-pages)，还查阅了Jekyll、kramdown等官网文档。找到了解决方法，在`_includes/head.html`的`<head> </head>`之间加入下列语句

```
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 

```

至于以上语句的写法，可以参看[MathJax官方文档](https://docs.mathjax.org)


于2018-07-20更新
