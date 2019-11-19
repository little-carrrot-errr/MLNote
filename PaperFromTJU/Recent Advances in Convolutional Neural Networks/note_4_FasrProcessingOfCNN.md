<!-- TOC -->

- [**Fast Processing of CNNs**](#fast-processing-of-cnns)
  - [**FFT**](#fft)
  - [**Structured Transforms**](#structured-transforms)
  - [**Low Precision**](#low-precision)
  - [**Weight Compression**](#weight-compression)
    - [**Vector Quantization(VQ)**](#vector-quantizationvq)
    - [**Pruning 剪枝**](#pruning-%e5%89%aa%e6%9e%9d)
    - [**Hash**](#hash)
  - [**Sparse Convolution**](#sparse-convolution)

<!-- /TOC -->

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

[张量分解](https://blog.csdn.net/yixianfeng41/article/details/73009210)

**自适应fastfood变换 \ Adaptive Fastfood transform**是对矩阵逼近的**Fastfood transform**[141]的推广。在全连接层的 **Adaptive Fastfood transformation**的权重矩阵表示如下:
> $Cx=(\widetilde{D_1}H\widetilde{D_2} \Pi H\widetilde{D_3})x$
>  
> $其中\widetilde{D_1}，\widetilde{D_2}以及\widetilde{D_3}都是参数的对角矩阵，\Pi是一个随机排列矩阵，H表示$**Walsh-Hadamard矩阵**

> 空间复杂度为$O(N)，时间复杂度为O(N Log N)。$

- 启发于基于循环矩阵在空间和计算效率方面的巨大优势[142，143]，Cheng等.[144]探讨全连通层参数化中的冗余性。 他们提出将循环结构施加在权矩阵上，以加快计算速度，并进一步允许使用FFT进行更快的计算。
- 循环矩阵$C \in R^{n×n}$作为全连通层参数矩阵，对每个输入向量$x\in R^n$ 有输出  $Cx\in R^{n\times n}$。这个输出可以用FFT高效计算并有$IFFT：CDx = ifft(fft(v)) \circ fft(x)。其中\circ代表元素乘法运算，v\in R^n 定义于C，D是改进模型容量的随机符号翻转矩阵$（**random sign flipping matrix**）。$该方法将时间复杂度从O(N^2)降低到O(N Log N)，空间复杂度从O(N^2)降到O(N)$。  
- Moczulski等人[145]通将对角矩阵与**正交离散余弦变换(DCT)**(**interleaving**、交叉？)，进一步推广循环结构，得到的变换$ACDC^{-1}具有O(N)空间复杂度和O(N Log N)时间复杂度$。

**[离散余弦变换](https://blog.csdn.net/li_wen01/article/details/72864485)**

---
## **Low Precision**

在CNN中，浮点数是一个神经元进行微小值更新的一个选择。但是这会是的参数包含了许多冗余的信息。为了减少冗余，**Binarized Neural Networks** （**BNNs 二值神经网络**）提出严格限制一些或者所有的算法包括输出都为二进制的值。

在神经网络中有三种二进制化（**binarization**）形式：二进制输入激活值，二进制二元突触权重，二进制输出激活值。完全二进制需要这三种成分都被二进制化，部分二进制被看做为部分二进制化（**partial binarization**）

<font size = 4px>**[二值神经网络参考](https://blog.csdn.net/lily_9/article/details/81409249)**</font>

二值化神经网络BNN有以下几个特点：
1. **减少内存占用**，权值和激活值二值化后，只需1bit即可表示，大幅度地降低了内存的占用。
2. **降低功耗**：因为二值化，原来32位浮点数，只需要1bit表示，存取内存的操作量降低了；其实，存取内存的功耗远大于计算的功耗，所以相比于32-bit DNN，BNN的内存占用和存取操作量缩小了32倍，极大地降低了功耗。
3. **减小面向深度学习的硬件的面积开销**：XNOR代替乘法，用1-bit的XNOR-gate代替了原来的32-bit浮点数乘法，对面向深度学习的硬件来说，有很大影响。譬如，FPGA，原来完成一个32-bit的浮点数乘法需要大约200个Slices，而1-bit 的Xnor-gate 只需要1个Slice，硬件面积开销，直接缩小了200倍。
4. **速度快**：paper作者专门写了一个二值乘法矩阵的GPU核，可以保证在准确率无损失的前提下，运行速度提高7倍。
5. [参考代码](https://github.com/MatthieuCourbariaux/BinaryNet)

【**paper**】
- Kim等人[147]考虑完全二值化，将某个突触（神经元）的一个预定部分设为零权重，其他所有的突触（**synapses**）权重都为1。他们的网络只需要 **$XNOR$**(异或非门)以及少量的计算操作。他们报告指出该网络在**MNIST**数据集上有98.7%的准确性。
- XNOR-Net[148]在Image Net数据集上应用卷积BNNs，其拓扑结构受AlexNet、ResNet和GoogLeNet的启发，报告阐明第一项-完全二值化的精度最高为51.2%，部分二值化为65.5%
-  DoReFa-Net [149]也在前后向反馈中减少了精度。在实验中探索了部分二值化和完全二值化，在Image Net上相应的top-1精度分别为43%和53%。
-   Courbariaux等人[150]描述了如何用完全二值化和批量归一化层训练完全连接的网络和cnn，并报告了MNIST、SVHN和CIFA R-10数据集的竞争准确性。

---
## **Weight Compression**
为了减少卷积层和全连通层中的参数，人们做了很多尝试。在这里，我们简要地介绍了在这些主题下的一些方法：矢量量化，剪枝和哈希。

### **Vector Quantization(VQ)**
<font size=3px>**[这里是一个VQ参考博客](https://blog.csdn.net/LG1259156776/article/details/52126697?utm_source=blogxgwz3)**</font><br>
[MATLAB代码](https://blog.51cto.com/12945177/1932205)

**矢量量化 VQ**是一种压缩密集连接的图层以使CNN模型更简单的方法。与标量量化相似，在标量量化中，大量的数字映射到较小的数字，VQ可以将数字组量化在一起，而不是一次寻址一组数字(addressing them one at time)。
<br>
[标量量化和矢量量化](https://blog.csdn.net/qingkongyeyue/article/details/52084012)
- 2013年，Denil等人[136]证明了神经网络参数中的冗余性，并利用VQ方法显着地减少了深度模型中动态参数的个数。
- Gong等人[152]研究了压缩CNN参数的信息论矢量量化方法，得到了与[136]相似的参数预测结果。他们还发现VQ方法比现有的矩阵因式分解方法有明显的改善。在VQ方法中，诸如乘积量化 **[product quantization](https://www.cnblogs.com/mafuqiang/p/7161592.html)** 等结构化量化方法的工作效率明显优于其他方法(如残差量化 **residual quantization**[153]、标量量化 **scalar quantization** [154])。 
- [高阶残差量化 HORQ](https://www.2cto.com/kf/201709/676972.html)

### **Pruning 剪枝**
它通过永久丢弃不太重要的连接来减少CNN中的参数和操作的数量[155]，从而使较小的网络能够从大型的前身网络继承知识。，并且工作并保持与性能相当的水平
- Han等人[146，156]通过基于量值(**magnitude-based**)的剪枝方法在网络中引入细粒度稀疏性。如果任何重量(权重？)的绝对大小小于一个标量阈值，则对该重量（权重？）进行剪枝
- [157]-扩展基于量值的方法，允许在以前的迭代中恢复被剪枝的权重，并通过紧密耦合的剪枝和再训练阶段来进行更大的模型压缩
- [158]考虑到权值之间的相关性，提出了一种能量感知的剪枝算法，该算法直接利用cnn的能耗估计来指导剪枝过程。

除了细粒度剪枝之外，还有一些研究粗粒度剪枝的工作:
- [159]建议删除经常在验证集上产生零输出激活的卷积核/filter
- [160]将类似的过滤器合并为一个，而Mariet等人[161]将具有类似输出激活的过滤器合并为一个。

### **Hash**
设计适当的散列技术来加速CNN的训练或节省内存空间也是一个有趣的问题。
- HashedNets[162]是最近的一项技术，通过使用哈希函数将连接权重分组到散列桶中，从而减少模型大小，并且在同一个散列桶中的所有连接共享一个等价物。它们的网络显著地缩小了神经网络的存储成本，而大部分保留了图像分类中的泛化性能。
- 如Shi等人[163]和Weinberger等人[164]所指出的稀疏性将使散列冲突最小化，从而使特征哈希更加有效。哈希网可以与剪枝一起使用，以更好地节省参数。

--- 
## **Sparse Convolution**
最近，许多尝试将卷积层的权重稀疏化-[165, 166]。
- [165]考虑基卷积核的稀疏表示，并通过利用卷积核的信道间和信道内冗余实现90%的稀疏化。
- 与稀疏化卷积层的权重不同，[166]提出 一种结构稀疏学习 **Structured Sparsity Learning (SSL)** [参考](https://blog.csdn.net/cookie_234/article/details/73195206)的方法同时优化它们的超参数(滤波器大小、深度和局部连通性)
- [167]提出了一种基于查找 **look-up based** 的卷积神经网络(**LCNN** [参考](https://blog.csdn.net/feynman233/article/details/69785592))，它通过很少的查找来编码一组丰富的字典，用于覆盖CNN中的权值空间。他们用字典和两个张量来解码卷积层的权重。diction nary被同一层中的所有权重过滤器之间共享，这允许CNN从很少的训练示例中学习。与标准的cnn相比，lcn可以在少量的迭代中获得更高的精度。

<br>

