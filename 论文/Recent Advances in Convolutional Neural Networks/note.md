# **Fast Processing of CNNs**
随着计算机视觉和机器学习任务的不断增加，深度神经网络的模型变得越来越复杂。这些强大的模型需要更多的数据来进行训练 以避免过度拟合。同时，海量的训练数据也带来了新的挑战，比如如何在可行的时间内对网络进行训练。

---
## **FFT**
Mathieu等人[49]用FFT在傅里叶域进行卷积运算。
[十分简明易懂的FFT（快速傅里叶变换）](https://blog.csdn.net/enjoy_pascal/article/details/81478582)。两个多项式相乘称为卷积，那么通过快速傅里叶，我们能快速求得多项式的点积。

使用FFT有很多优势：
1. 首先，傅里叶变换的卷积核可以重复使用，因为卷积核在一个mini-batch中与多幅图像相卷积
2. 傅里叶变化得到的输出的梯度在反向传播中不但可以用于输入图像，也能用来更新卷积核
3. 输入通道channel上的求和可以在傅里叶域中进行，这样每幅图像的输出通道只需要一次逆傅里叶变换。

然而，使用FFT进行卷积需要额外的内存来将特征映射存储在傅里叶域中，因为卷积核必须填充到与输入相同的大
小。当步长参数大于1时，空间花销尤其昂贵，这在许多最先进的网络中是常见的，例如[132]和[10]中的早期层。

虽然FFT可以实现更快的训练和测试过程，但呼声越来越小高的型卷积器已经成为CNN中的一个重要组成部分，如ResNet[12]和Google网[10]， 这使得一种专门针对小过滤器大小的新方法被提出： **Winograd’s minimal filtering algorithms [133]。**Winograd类似fft，在应用逆变换之前，减少transform space中的跨信道卷积，从而提高推理inference的效率。

**[抽空自己做做：卷积和快速傅里叶变换（FFT）的实现](https://blog.csdn.net/whjkm/article/details/81949356)**

---
## **Structured Transforms**

在各种文章中低阶矩阵分解被多次提出来用作优化问题。矩阵分分解：给定一个$m \times n的矩阵C，其秩为r$。那么存在一个等式$C=AB，其中A是m\times r的列满秩矩阵，B是r\times n的行满秩矩阵$。为了一个分数$p$上减少C的参数，满足条件$mr + rn < pmn$是十分重要的，即C的秩应该满足$r < pmn/(m+n)$。应用该因式分解，空间复杂度由O(mn)降为O(r(m+n))，时间复杂度由O(mn)降至O(r(m+n))。
- Sainath等人[134]将低阶矩阵因式分解应用于深度cnn中的最终权层，训练时间加快了30-50%，且精度损失很小。
- 同样，Xue等人[135]在深度CNN的每一层上应用奇异值分解，使模型尺寸减少71%，相对精度损失小于1%。
- Denton[137]和Jaderberg等人[138]在[136]的启发下证明了深层神经网络中卷积核的冗余性，并提出了近似器 **approximations** 减少所需的计算量。
-  Novikov 等人[139]推广低秩思想，将权矩阵看作多维张量，并应用张量列分解 **Tensor-Train decomposition** [140]来减少全连通的参数

**自适应fastfood变换 \ Adaptive Fastfood transform**是对矩阵逼近的**Fastfood transform**[141]的推广。在全连接层的 **Adaptive Fastfood transformation**的权重矩阵表示如下:
> $Cx=(\widetilde{D_1}H\widetilde{D_2} \Pi H\widetilde{D_3})x$
>  
> $其中\widetilde{D_1}，\widetilde{D_2}以及\widetilde{D_3}都是参数的对角矩阵，\Pi是一个随机排列矩阵，H表示$**Walsh-Hadamard矩阵**

> 空间复杂度为$O(N)，时间复杂度为O(N Log N)。$

- 启发于基于循环矩阵在空间和计算效率方面的巨大优势[142，143]，Cheng等.[144]探讨全连通层参数化中的冗余性。 他们提出将循环结构施加在权矩阵上，以加快计算速度，并进一步允许使用FFT进行更快的计算。
- 循环矩阵$C \in R^{n×n}$作为全连通层参数矩阵，对每个输入向量$x\in R^n$ 有输出  $Cx\in R^{n\times n}$。这个输出可以用FFT高效计算并有$IFFT：CDx = ifft(fft(v)) \circ fft(x)。其中\circ代表元素乘法运算，v\in R^n 定义于C，D是改进模型容量的随机符号翻转矩阵$（**random sign flipping matrix**）。$该方法将时间复杂度从O(N^2)降低到O(N Log N)，空间复杂度从O(N^2)降到O(N)$。  
- Moczulski等人[145]通将对角矩阵与**正交离散余弦变换(DCT)**(**interleaving**、交叉？)，进一步推广循环结构，得到的变换$ACDC^{-1}具有O(N)空间复杂度和O(N Log N)时间复杂度$。

**[离散余弦变换](https://blog.csdn.net/li_wen01/article/details/72864485)**