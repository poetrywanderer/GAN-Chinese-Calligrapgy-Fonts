# GAN on Chinese Calligraphy Fonts

[TOC]

## Basic knowledge

> **说明：**因为网上已经有许多优秀的理论介绍文档，想系统学习的可以直接跳到本文的**References**部分，本文仅从实践角度对必要理论进行总结。

### 1. 生成对抗

下图很好地说明了生成对抗模型的思想，其实就是生成器与判别器之间的相互博弈，最终使得生成器学习到真实数据的分布特征，达到 ”以假乱真“ 的效果。

![GAN](./pics for blog/GAN.jpg)

### 2. 生成模型和判别模型

生成模型和判别模型的设计无疑是最关键的部分。区别在于生成器需要学习真是样本的所有特征，而判别器只需要学习到两者的不同特征即可。但往往的做法是将两种网络设计成对称的结构，如GAN, DCGAN等。下面是某博主按顺序总结的2014~2019.3之间的所有论文，详见Reference. 1。

![GAN模型总结](./pics for blog/GAN模型.png)

### 3. 目标函数

由于生成对抗模型本质上还是两个神经网络，需要进行反向传播来更新权重，因此需要设计目标函数。GAN原文给出的目标函数如下：

![](./pics for blog/math.svg)

其中，Pdata(x) 为真样本的分布，Pz(z) 为假样本的分布，E 为期望。这里其实是两个网络损失函数的组合，分开看就好：

![](./pics for blog/优化D.jpg)

优化 D，即优化判别网络时，没有生成网络什么事，x 为真样本，后面的 G(z) 就相当于已经得到的假样本，因此这里就相当于一个**二分类**的交叉熵损失函数。优化 D 的公式的第一项，使得真样本 x 输入的时候，得到的结果越大越好，因为真样本的预测结果越接近1越好；对于假样本G(z)，需要优化的是其结果越小越好，也就是D(G(z)) 越小越好，因为它的标签为 0。但是第一项越大，第二项越小，就矛盾了，所以把第二项改为 1-D(G(z))，这样就越大越好。

![](./pics for blog/优化G.jpg)

优化G，只与第二项有关，这个时候希望假样本的标签是 1，所以是 D(G(z))越大越好，但是为了统一成 1-D(G(z))的形式，那么只能是最小化 1-D(G(z))，本质上没有区别，只是为了形式的统一。之后这两个优化模型可以合并起来写，就变成最开始的最大最小目标函数了。事实上，生成器的目标函数等价于优化真样本分布和假样本分布之间的 **JS 散度**，即分布的相似性（有兴趣可以自己查）。

需要注意的是，实际训练时，采用交替训练，即先训练D，后训练G，不断重复。因此，对于生成器，其最小化的是—判别器目标函数的最大值。

实际上，衡量两个分布之间距离的方式有很多种，后来的学者通过定义不同的度量方式，提出了LSGAN，EBGAN等模型来改进GAN训练的稳定性。

### 4. 训练效果评价

前面说到，训练过程中会产生两个变化相反的loss曲线，因此无法根据loss的变化来判断模型的训练效果。除了用肉眼进行粗略判断外，常见的有Inception Score，FID等。详见Reference 2.

### 5. References

1. [Blog:【学习笔记】生成模型——生成对抗网络](http://www.gwylab.com/note-gans.html)
2. [Blog: GAN万字长文综述(郭晓锋)](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/88684919)
3. [Github: DeepLearning-500-questions](https://github.com/scutan90/DeepLearning-500-questions/tree/master/ch07_%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C(GAN))
4. [Paper: How Generative Adversarial Networks and its variants Work: An Overview of GAN](https://arxiv.org/pdf/1711.05914.pdf)

## Example work: GAN on Chinese Calligraphy Fonts

介绍完理论，本节介绍一下最近的工作：利用GAN生成中国书法汉字。（左：生成图；中：过程图；右：原图）

​             ![](./pics for blog/Fake_c_SNGAN.png)       ![](./pics for blog/generate_animation.gif)       ![](./pics for blog/Real_c_SNGAN.png) 

### 1. SNGAN

待补充

### References

1. SNGAN <https://github.com/crcrpar/pytorch.sngan_projection>
2. DRAGAN





