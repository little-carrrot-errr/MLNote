<!-- TOC -->

- [**Convolutional Layer**](#convolutional-layer)
  - [**广义线性模型**](#%e5%b9%bf%e4%b9%89%e7%ba%bf%e6%80%a7%e6%a8%a1%e5%9e%8b)
  - [**关于 CNN对图像特征的 位移、尺度、形变不变性的理解**](#%e5%85%b3%e4%ba%8e-cnn%e5%af%b9%e5%9b%be%e5%83%8f%e7%89%b9%e5%be%81%e7%9a%84-%e4%bd%8d%e7%a7%bb%e5%b0%ba%e5%ba%a6%e5%bd%a2%e5%8f%98%e4%b8%8d%e5%8f%98%e6%80%a7%e7%9a%84%e7%90%86%e8%a7%a3)
  - [**shit-invariance $\quad$ 参考视频**](#shit-invariance-quad-%e5%8f%82%e8%80%83%e8%a7%86%e9%a2%91)
  - [**平铺卷积网络 Tiled Convolution** $\quad$参考1 && 2](#%e5%b9%b3%e9%93%ba%e5%8d%b7%e7%a7%af%e7%bd%91%e7%bb%9c-tiled-convolution-quad%e5%8f%82%e8%80%831--2)
  - [**图像卷积与反卷积 Transposed Convolution $\quad$ CNN中的卷积和反卷积**](#%e5%9b%be%e5%83%8f%e5%8d%b7%e7%a7%af%e4%b8%8e%e5%8f%8d%e5%8d%b7%e7%a7%af-transposed-convolution-quad-cnn%e4%b8%ad%e7%9a%84%e5%8d%b7%e7%a7%af%e5%92%8c%e5%8f%8d%e5%8d%b7%e7%a7%af)
  - [**空洞卷积 Dilated Convolution**](#%e7%a9%ba%e6%b4%9e%e5%8d%b7%e7%a7%af-dilated-convolution)
  - [**Network In Network——卷积神经网络的革新**](#network-in-network%e5%8d%b7%e7%a7%af%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c%e7%9a%84%e9%9d%a9%e6%96%b0)
  - [**[ Inception Module / GoogleNet]()**](#inception-module--googlenet)
  - [**1×1卷积**](#1%c3%971%e5%8d%b7%e7%a7%af)
  - [**Inception Module**](#inception-module)
- [**Pooling Layer**](#pooling-layer)
  - [**L_p Pooling**](#lp-pooling)
  - [<b>Mixed Pooling</b>](#bmixed-poolingb)
  - [<b>Stochastic Pooling</b>](#bstochastic-poolingb)
  - [<b>Spectral Pooling</b>](#bspectral-poolingb)
  - [**Spatial Pyramid Pooling**](#spatial-pyramid-pooling)
  - [<b>Multi-scale Orderless Pooling</b>](#bmulti-scale-orderless-poolingb)
  - [**VLAD && BoW && Fisher Vector && Global CNN**](#vlad--bow--fisher-vector--global-cnn)
- [**Activation Function**](#activation-function)
  - [**激活函数的饱和性**](#%e6%bf%80%e6%b4%bb%e5%87%bd%e6%95%b0%e7%9a%84%e9%a5%b1%e5%92%8c%e6%80%a7)
  - [**ReLU**](#relu)

<!-- /TOC -->

## **Convolutional Layer**
### **[广义线性模型](https://blog.csdn.net/weixin_37140379/article/details/82289704)**

### **[关于 CNN对图像特征的 位移、尺度、形变不变性的理解](https://blog.csdn.net/voxel_grid/article/details/79275637)**

### **[shit-invariance](https://www.cnblogs.com/fydeblog/p/11083664.html) $\quad$  [参考视频](https://www.bilibili.com/video/av63925068)**
  - 什么是平移等方差（Shift-equivariance）？ [参考](https://www.cnblogs.com/fydeblog/p/11083664.html)
    >答：$Shift {\Delta h, \Delta w}(\widetilde{\mathcal{F}}(X))=\widetilde{\mathcal{F}}\left(\text { Shift }{\Delta h, \Delta w}(X)\right) \quad \forall(\Delta h, \Delta w)$，可以看到输入在$(\Delta h, \Delta w)$变化，输出对应的输出在$(\Delta h, \Delta w)$变化。

  - 什么是平移不变性（Shift-invariance）？

    >答：$\widetilde{\mathcal{F}}(X)=\widetilde{\mathcal{F}}\left(\text { Shift }_{\Delta h, \Delta w}(X)\right) \quad \forall(\Delta h, \Delta w)$， 输入在$(\Delta h, \Delta w)$变化，不改变最后的结果。

    大多数现代的卷积网络是不具有平移不变性的（如上所示，右边是作者提出的方法BlurPool），而不具有平移不变性的原因是因为maxpooling，strided-convolution以及average-pooling这些下采样方法忽略了抽样定理
    

### **平铺卷积网络 Tiled Convolution** $\quad$[参考1](https://blog.csdn.net/xiao_jiang2012/article/details/9349955) && [2](https://blog.csdn.net/zhq9695/article/details/84959472)

###  **[图像卷积与反卷积 Transposed Convolution](https://blog.csdn.net/qq_38906523/article/details/80520950) $\quad$ [CNN中的卷积和反卷积](https://blog.csdn.net/sinat_29957455/article/details/85558870)**
  
###  **[空洞卷积 Dilated Convolution](https://www.jianshu.com/p/f743bd9041b3)**

### **[Network In Network——卷积神经网络的革新](https://www.cnblogs.com/yinheyi/p/6978223.html)**
  - [1 × 1 卷积](https://blog.csdn.net/renhaofan/article/details/82721868s)

### **[ Inception Module / GoogleNet]()**
###  **1×1卷积**
   -  作用1：在相同尺寸的感受野中叠加更多的卷积，能提取到更丰富的特征。这个观点来自于Network in Network(NIN, https://arxiv.org/pdf/1312.4400.pdf)，图1里三个1x1卷积都起到了该作用
   -  作用2：使用1x1卷积进行降维，降低了计算复杂度。

### **[Inception Module](https://www.cnblogs.com/leebxo/p/10315490.html)**
  - 多个尺寸上进行卷积再聚合（提取不同尺度特征、稀疏矩阵分解-1×1、3×3、5×5进行特征维度分解、利用1×1...把各个维度相关性强的特征汇聚）
  - 密集计算子结构组合而成的稀疏模块来用于特征提取及表达
  - 通过一种spared layer architecture来实现较优的多维度特征表达，然后通过对这种结构进行叠加，中间不时再插入一些MaxPool层以减少参数数目（从而节省内存与计算开销），最终就行成了Inception v1分类模型。[参考](https://www.jianshu.com/p/57cccc799277)
  - 借鉴Network in Network中的idea,在每个子conv层里使用了1x1的conv来作上一层的输入feature maps的channels数缩减、归总。
    ![](\inception&#32;module1.png)
  - 模型的最后会选通过一个AvgPool层来处理最终的feature maps，然后再由FC层汇总生成1000个输出，进而由Softmax来得到1000类的概率分布

## **Pooling Layer**

### **L_p Pooling**
  
  模型启发于复杂细胞。有相关论文表明$L_p Pooling$提供比$max Pooling$更好的效果。
  公式：
  <font size=4px>$y_{i,j,k} = [ \sum_{(m,n) \in R_{i,j}} (a_{m,n,k})^p ]^{1/p}$</font>
  
  当p=1时，此方法等于average pooling，；当$p=\infty$时，该方法等价于max pooling
### <b>Mixed Pooling</b>
  
   <font size=4px>$y_{i,j,k} = \lambda \max_{(m,n)\in R_{i,j}} a_{m,n,k} +(1-\lambda) \frac{1}{|R_{ij}|} \sum_{(m,n \in R_{i,j})} a_{m,n,k}$</font>

   $\lambda$为0或者1，当$\lambda$为1时，该方法为max pooling；当$\lambda$为0时，该方法average pooling；在前向传播过程中，$\lambda$值会被记录下来，以便后期后向传播计算使用。Experiments in [46] show that mixed pooling can better address the overfitting problems and it
performs better than max pooling and average pooling.

### <b>Stochastic Pooling</b>
    
    首先通过归一化区域内的激活函数来计算每个区域j的概率p：
    ><font size=4px> $p_i = \frac{a_i}{\sum_{k \in R_j} a_k}$</font>

    然后基于p的多项式分布，在区域中抽取一个位置l。合并的激活仅仅是$a_l : s_j =a_l \quad l\sim P(p_1,...,p_{|R_j|})\quad$ ，过程如下图: ![](/stochastic_pooling.jpg)
    随机池化具有最大池化的优点（消除非极大值，降低了上层的计算复杂度），同时由于随机性它能够避免过拟合。

### <b>Spectral Pooling</b>
    
    通过裁剪输入频域的表达来减少维度。
    对于输入特征映射$x \in R^{m × m}$，假设期望输出为$h × w$：
    1. 计算特征映射的离散傅里叶变换
    2. 然后通过只保持频率的中心$h×w$子矩阵来实现频率表示，
    3. 使用IDFT将近似值映射回空间域

    与最大池化相比：
    - 通过低通filtering操作，对于同样的输出，谱池化能保留更多信息
    - 同时，它也不受其他池方法所显示的输出映射维数急剧下降的影响
    - 此外，频谱池的过程是通过矩阵截断实现的，这使得它能够在使用FFT【快速傅里叶变换 (fast Fourier transform)】处理卷积核的CNNs中以很少的计算成本实现。
### **Spatial Pyramid Pooling**
    
    [参考](https://www.cnblogs.com/zongfa/p/9076311.html)
    在一般的CNN结构中，在卷积层后面通常连接着全连接。而全连接层的特征数是固定的，所以在网络输入的时候，会固定输入的大小(fixed-size)。但在现实中，我们的输入的图像尺寸总是不能满足输入时要求的大小。然而通常的手法就是裁剪(crop)和拉伸(warp)。

    图像的纵横比(ratio aspect) 和 输入图像的尺寸是被改变的。这样就会扭曲原始的图像。而Kaiming He在这里提出了一个SPP(Spatial Pyramid Pooling)层能很好的解决这样的问题， 但SPP通常连接在最后一层卷基层。![](/spatial_pyramid_pooliing.png)

    **SPP 显著特点**
  1) 不管输入尺寸是怎样，SPP 可以产生固定大小的输出 
  2) 使用多个窗口(pooling window)，窗口的数量取决于输入的大小
  3) SPP 可以使用同一图像不同尺寸(scale)作为输入, 得到同样长度的池化特征。
  4) SPP 对于特定的CNN网络设计和结构是独立的。(也就是说，只要把SPP放在最后一层卷积层后面，对网络的结构是没有影响的， 它只是替换了原来的pooling层) 
   ![](/Spatial_Pyramid_Pooling_Layer.png)

    注意我们上面曾提到使用多个窗口(pooling窗口，上图中蓝色，青绿，银灰的窗口， 然后对feature maps 进行pooling），将分别得到的结果进行合并就会得到固定长度的输出), 这就是得到固定输出的秘密原因。  

### <b>Multi-scale Orderless Pooling</b>
  
  [参考1](https://blog.csdn.net/qq_32417287/article/details/80372422)
  [参考2](https://blog.csdn.net/happyer88/article/details/51418059)
  - 解决问题：在保证CNN特征的区分能力的同时，提高CNN特征的对各种不变性的鲁棒性。
  - 解决思路：利用多尺度无序池(MOP)在不降低CNNs区分能力的前提下，提高了CNNs的不变性。在不同尺度上的局部 patch 上提取CNN特征，在不同尺度上分别对其进行无序的VLAD池化操作，将不同尺度上的池化之后的特征连接起来形成最终的特征
  - 为什么滑窗呢？
    如果region取得稍有不对（离目标有所偏差），根据patch CNN预测的图像label也会错的离谱。

### **VLAD && BoW && Fisher Vector && Global CNN**



## **Activation Function**
### **激活函数的饱和性**


### **ReLU**

