<!-- TOC -->

- [**CNN经典模型整理Lenet，Alexnet，Googlenet，VGG，Deep Residual Learning,squeezenet**](#cnn%e7%bb%8f%e5%85%b8%e6%a8%a1%e5%9e%8b%e6%95%b4%e7%90%86lenetalexnetgooglenetvggdeep-residual-learningsqueezenet)
- [**LeNet-5**](#lenet-5)
- [**卷积神经网络之AlexNet**](#%e5%8d%b7%e7%a7%af%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c%e4%b9%8balexnet)
- [**ZFNet**](#zfnet)
- [**VGG Net**](#vgg-net)
- [**GoogLeNet**](#googlenet)
  - [**Inception V1**](#inception-v1)
  - [**Inception V2**](#inception-v2)
  - [**Inception V3**](#inception-v3)
  - [**Inception V4**](#inception-v4)
- [**深度残差网络（DRN） ResNet**](#%e6%b7%b1%e5%ba%a6%e6%ae%8b%e5%b7%ae%e7%bd%91%e7%bb%9cdrn-resnet)
  - [**问题：反向传播过程中的梯度消失问题**](#%e9%97%ae%e9%a2%98%e5%8f%8d%e5%90%91%e4%bc%a0%e6%92%ad%e8%bf%87%e7%a8%8b%e4%b8%ad%e7%9a%84%e6%a2%af%e5%ba%a6%e6%b6%88%e5%a4%b1%e9%97%ae%e9%a2%98)
  - [**深度残差网络（Deep Residual Network，简称DRN）**](#%e6%b7%b1%e5%ba%a6%e6%ae%8b%e5%b7%ae%e7%bd%91%e7%bb%9cdeep-residual-network%e7%ae%80%e7%a7%b0drn)
- [**全文知识结构**](#%e5%85%a8%e6%96%87%e7%9f%a5%e8%af%86%e7%bb%93%e6%9e%84)

<!-- /TOC -->
# **CNN经典模型整理Lenet，Alexnet，Googlenet，VGG，Deep Residual Learning,squeezenet**
[参考](https://blog.csdn.net/m0_37264397/article/details/75174484)


# **LeNet-5**
[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)

[以下资料参考](https://blog.csdn.net/m0_37749527/article/details/79248187)

我们先来简单浏览一遍整个模型所使用的的计算方法：

1. **基于梯度的学习（梯度下降）：**
   
   > <font size=4px>$W_k=W_{k-1} - \epsilon \frac{\partial E(W)}{\partial W}$</font>
2. **反向传播:**
   
   gradients can be computed efficiently by propagation from the output to the input 对误差进行反向传播，更新权值
   > ![](\img/lenet-5/backpropagation.png)
   
   Xn is a vector representing the **output** of the module. Wn is the vector of **tunable parameters** in the module a subset of W and Xn is the module’s **input** vector as well as the previous module’s output vector

3. **feature maps**

    这个单词国人把它翻译成特征图，挺起来很专业的名词。那么什么叫特征图呢？特征图其实说白了就是CNN中的每张图片，都可以称之为特征图张。在CNN中，我们要训练的卷积核并不是仅仅只有一个，这些卷积核用于提取特征，卷积核个数越多，提取的特征越多，理论上来说精度也会更高，然而卷积核一堆，意味着我们要训练的参数的个数越多。在LeNet-5经典结构中，第一层卷积核选择了6个，而在AlexNet中，第一层卷积核就选择了96个，具体多少个合适，还有待学习。

    回到特征图概念，CNN的每一个卷积层我们都要人为的选取合适的卷积核个数，及卷积核大小。每个卷积核与图片进行卷积，就可以得到一张特征图了，比如LeNet-5经典结构中，第一层卷积核选择了6个，我们可以得到6个特征图，这些特征图也就是下一层网络的输入了。我们也可以把输入图片看成一张特征图，作为第一层网络的输入。

4. **Convolutional Networks**

    Convolutional Networks combine three architecturalideas to ensure some degree of shift, scale and distortion invariance: **local receptive fields**,**shared weights** (or weight replication) and spatial or temporal **sub-sampling**

    卷积网络的三个要点：局部感受野、权值共享、下采样

    - **localreceptive fields** : Each unit in a layer receives inputs from a set of units located in a small neighborhood in the previous layer.  局部感受野，每个局部单元共享权值。

    - **feature map** : Units in a layer are organized in planes within which all the units share the same set of weights. The set of outputs of the units in such a plane is called a feature map. 共享权值的各局部单元输出形成一个feature map。

    - **sub-sampling** : The receptive field of each unit is a 2 by 2 area in the previous layer’s corresponding feature map. Units are non-overlapping. sub-sampling performs a local averaging and reduces the spatial solution of the feature map. 下采样，减小卷积层的尺寸，通过求局部平均降低特征图的分辨率，并且降低了输出对平移和形变的敏感度。
    
        缩小图像（或称为下采样（subsampled）或降采样（downsampled））的主要目的有两个：1、使得图像符合显示区域的大小；2、生成对应图像的缩略图

         下采样原理：对于一副图像I尺寸为$M*N$，对起进行s倍下采样，即得到$（M/s）*（N/s）$尺寸的分辨率图像，当然，s应该是M和N的公约数才可以，如果考虑是矩阵形式的图像，就是把原始图像s*s窗口内的图像变成一个像素，这个像素点的值就是窗口内所有像素的均值。<font size=4.5px>$P_k = \sum \frac{I_i}{s^2}$</font>
5.  **Loss Function**
    - Maximum Likelihood Estimation criterion (MLE)
        
        ![](\img/lenet-5/MLE.png) 
        
        [极大似然估计详解](https://blog.csdn.net/qq_39355550/article/details/81809467)
    - Maximum a posteriori criterion (MAP) 
        
        ![](\img/lenet-5/loss2.jpg)

        posterior ∝ likelihood × prior (Beyasian Theory)
        [最大后验估计Maximum-a-Posteriori Estimation](https://www.cnblogs.com/easoncheng/archive/2012/11/08/2760675.html)
    
    损失函数：损失函数最小相当于似然函数取得最大值

    贝叶斯方法：求后验最大似然函数

6. **LeNet-5网络结构**
    ![](\img/lenet-5/classical_lenet5.png)

    输入为$32 \times 32$像素的图像。从图像中看出来我们一共有7层神经层：
    - **C1**：$5\times5$的大小，一共6层卷积层，最后得的 $(32-(5-1))=28$，大小为$28 \times 28$的特征映射。
      - 其中我们一共有$(5\times5+1)\times6=156$个参数，计算次数为$(5\times5+1)\times28\times28\times6=122304$次
    - **S2**：$2\times2$的大小，一共6层卷积层，下采样，得到$(28/2=14)$，大小为14×14的特征映射。对输入进行$2\times2=4$个像素进行下采样，然后将得到的结果乘于一个可训练权重，最后加上一个可训练的偏置。
      - 那么我们有$(1+1)\times6=12$个训练参数；计算次数为$(2\times2+1)\times14\times14\times6=5880$
    - **C3**：$5\times5$的大小，一共16层卷积层，最后得的 $(14-(5-1))=10$，大小为$10 \times 10$的特征映射。C3的每个feature map并不与S2所有feature map 相连接。 不使用全连接的原因有两方面。首先，非完整连接方案将连接数量保持在合理范围内。更重要的是，它打破了网络的对称性。不同的特征映射被迫提取不同的(希望是互补的)特征，因为它们得到不同的输入集。
        ![](\img/lenet-5/table1.png)
        
        表1中的连接方案背后的原理如下。前6个C3的feature map与S2相邻的3个feature map。接下来的6个与S2的4个相邻feature map相连。接下来的3个是与不相邻的四个feature map相连。最后，最后一个与S2 所有feature map相连。
      - 参数：$(5*5\times3+1)\times6+(5*5\times4+1)\times9+(5*5\times6+1)=1516$；计算次数：$1516×10×10=151600$
    - **S4**：$2\times2$输入大小，共16层特征提取层，这一层作为下采样层。输出大小为$10/2=5$，即$5\times5$。
      - 参数：$2 \times 16 =32$；计算次数$(2\times2+1)\times5\times5\times16=2000$
    - **S5**：大小为$5×5$, 共120 层feature maps的卷积层，输出结果为$(5-(5-1))=1$，$1\times1$大小的特征矩阵，即与**S4**全连接。C5被标记为一个卷积层，而不是一个完全连接的层，因为如果LeNet-5的输入在保持不变的情况下变大，feature map维度将大于1x1。
       - 训练次数：$(5×5×16+1)×120=48120$
     - **F6**：全连接层，一共84个节点，每个大小为120。先计算与上一层点积，加上bias，再传入sigmoid函数。训练参数：$(120+1)×84=10164$。在这个部分采用了正切函数：<font size=4px>$x_i = f(a_i)=A\tanh(Sa_i)$</font>
     - **output layer**: Euclidean RadialBasis Function units (RBF) for each class；输出层，每类一个输出，输出该类对应的**RBF** ： <font size=4px>$y_j=\sum_j (x_i - w_{ij})^2$</font>

> **Radial basis function** 指的是我们要计算的函数的结果只和距离(∥x−xn∥)有关；

> **Radial basis function**：径向基函数是一个取值仅仅依赖于离原点距离的实值函数。也就是Φ(x)=Φ(‖x‖)，或者还可以是到任意一点cc的距离，cc点称为中心点，也就是Φ(x,c)=Φ(∥x−c∥)

[LeNet简单代码1](https://blog.csdn.net/u012897374/article/details/78575594)

[Lenet代码参考2](https://blog.csdn.net/Rasin_Wu/article/details/79935952)

<br>


# **卷积神经网络之AlexNet**

[参考自此博客](https://www.cnblogs.com/wangguchangqing/p/10333370.html)
  
- 使用RELU作为激活函数
- 数据增强
- 层叠池化
- 局部相应归一化
- Dropout
- AlexNet网络结果
  
    ![](\../img/note1/AlexNet.png)

    网络包含8个带权重的层；前5层是卷积层，剩下的3层是全连接层。最后一层全连接层的输出是1000维softmax的输入，softmax会产生1000类标签的分布网络包含8个带权重的层；前5层是卷积层，剩下的3层是全连接层。最后一层全连接层的输出是1000维softmax的输入，softmax会产生1000类标签的分布。
    - **卷积层C1**
      
      该层的处理流程是： 卷积-->ReLU-->池化-->归一化
        - 卷积：输入是$227×227$，使用96个$11×11×3$的卷积核，得到的FeatureMap为$55×55×96$
        - ReLU：将卷积层输出的FeatureMap输入到ReLU函数中
        - 池化：使用3×3步长为2的池化单元（重叠池化，步长小于池化单元的宽度），输出为$27×27×96 。 (55−3)/2+1=27$
        - 局部响应归一化：使用$k=2,n=5,α=10−4,β=0.75$进行局部归一化，输出的仍然为$27×27×96$，输出分为两组，每组大小为$27 \times 27 \times 48$
    - **卷积层C2**      
      该层的处理流程是： 卷积-->ReLU-->池化-->归一化
      - 卷积：输入是2组$27×27×48$。使用2组，每组128个尺寸为$5×5×48$的卷积核，并作了边缘填充$padding=2$，卷积的步长为1. 则输出的FeatureMap为2组，每组的大小为$27×27 \times128$。 $(27+2∗2−5)/1+1=27$
      - ReLU：将卷积层输出的FeatureMap输入到ReLU函数中
      - 池化：使用3×3步长为2的池化单元（重叠池化，步长小于池化单元的宽度），输出为$13×13×256 。  (27−3)/2+1=13$
      - 局部响应归一化：使用$k=2,n=5,α=10−4,β=0.75$进行局部归一化，输出的仍然为$13×13×256$，分为两组，每组大小为$13×13×128$
    - **卷积层C3**
        
        该层的处理流程是： 卷积-->ReLU
      - 卷积：输入是$13×13×256$，使用2组共384尺寸为$3×3×256$的卷积核，做了边缘填充$padding=1$，卷积的步长为1.则输出的FeatureMap为$13×13 \times384$
      - ReLU：将卷积层输出的FeatureMap输入到ReLU函数中
    - **卷积层C4**
        
        该层的处理流程是： 卷积-->ReLU。该层和C3类似。
      - 卷积：输入是$13×13×384$，分为两组，每组为$13×13×192$.使用2组，每组192个尺寸为$3×3×192$的卷积核，做了边缘填充$padding=1$，卷积的步长为1.则输出的FeatureMap为$13×13 \times384$，分为两组，每组为$13×13×192$
      - ReLU：将卷积层输出的FeatureMap输入到ReLU函数中
      
    - **卷积层C5**

        该层处理流程为：卷积-->ReLU-->池化
      - 卷积：输入为$13×13×384$，分为两组，每组为$13×13×192$。使用2组，每组为128尺寸为$3×3×192$的卷积核，做了边缘填充padding=1，卷积的步长为1.则输出的FeatureMap为$13×13×256$
      - ReLU：将卷积层输出的FeatureMap输入到ReLU函数中
      - 池化：池化运算的尺寸为$3×3$，步长为2，池化后图像的尺寸为 $(13−3)/2+1=6$,即池化后的输出为$6×6×256$

    - **全连接层FC6**

        该层的流程为：（卷积）全连接 -->ReLU -->Dropout
      - 卷积->全连接： 输入为$6×6×256$,该层有$4096$个卷积核，每个卷积核的大小为$6×6×256$。由于卷积核的尺寸刚好与待处理特征图（输入）的尺寸相同，即卷积核中的每个系数只与特征图（输入）尺寸的一个像素值相乘，一一对应，因此，该层被称为全连接层。由于卷积核与特征图的尺寸相同，卷积运算后只有一个值，因此，卷积后的像素层尺寸为$4096×1×1$，即有4096个神经元。
      - ReLU：这4096个运算结果通过ReLU激活函数生成4096个值
      - Dropout：抑制过拟合，随机的断开某些神经元的连接或者是不激活某些神经元
      
    - **全连接层FC7**

        流程为：全连接-->ReLU-->Dropout
      - 全连接：输入为4096的向量
      - ReLU：这4096个运算结果通过ReLU激活函数生成4096个值
      - Dropout：抑制过拟合，随机的断开某些神经元的连接或者是不激活某些神经元

    -  **输出层**
       
       第七层输出的4096个数据与第八层的1000个神经元进行全连接，经过训练后输出1000个float型的值，这就是预测结果。

- **AlexNet参数数量**
  >卷积层的参数 = 卷积核的数量 * 卷积核 + 偏置
  - **C1：** 96个11×11×3的卷积核，96×11×11×3+96=34848
  - **C2**：**2组**，每组128个5×5×48的卷积核，(128×5×5×48+128)×2=307456
  - **C3**：384个3×3×256的卷积核，3×3×256×384+384=885120
  - **C4**： **2组**，每组192个3×3×192的卷积核，(3×3×192×192+192)×2=663936
  - **C5**：**2组**，每组128个3×3×192的卷积核，(3×3×192×128+128)×2=442624
  - **FC6**： 4096个6×6×256的卷积核，6×6×256×4096+4096=37752832
  - **FC7**： 4096∗4096+4096=16781312
  - **Output**: 4096∗1000=4096000
  - 卷积层 C2，C4，C5中的卷积核只和位于同一GPU的上一层的FeatureMap相连。从上面可以看出，参数大多数集中在全连接层，在卷积层由于权

<br>


# **ZFNet**
> 该论文是在AlexNet基础上进行了一些细节的改动，网络结构上并没有太大的突破。该论文最大的贡献在于通过使用可视化技术揭示了神经网络各层到底在干什么，起到了什么作用。

> s使用一个多层的反卷积网络来可视化训练过程中特征的演化及发现潜在的问题；同时根据遮挡图像局部对分类结果的影响来探讨对分类任务而言到底那部分输入信息更重要。

[参考1](https://blog.csdn.net/cdknight_happy/article/details/78855172)

[参考2](https://blog.csdn.net/chenyuping333/article/details/82178769)

 <br>


# **VGG Net**
[参考](https://www.cnblogs.com/zhenggege/p/9000414.html)

![](\../img/note1/VGG.jpg)

1. 选择采用$3*3$的卷积核是因为$3*3$是最小的能够捕捉像素8邻域信息的的尺寸。
2. 使用1*1的卷积核目的是在不影响输入输出的维度情况下，对输入进行形变，再通过ReLU进行非线性处理，提高决策函数的非线性。
3. 2个$3*3$卷积堆叠等于1个$5*5$卷积，3个$3*3$堆叠等于1个$7*7$卷积，感受野大小不变，而采用更多层、更小的卷积核可以引入更多非线性（更多的隐藏层，从而带来更多非线性函数），提高决策函数判决力，并且带来更少参数。
4. 每个VGG网络都有3个FC层，5个池化层，1个softmax层。
5. 在FC层中间采用dropout层
6. 训练采用多尺度训练（Multi-scale），将原始图像缩放到不同尺寸 S，然后再随机裁切224*224的图片，并且对图片进行水平翻转和随机RGB色差调整，这样能增加很多数据量，对于防止模型过拟合有很不错的效果。

<br>


# **GoogLeNet**

[参考](https://my.oschina.net/u/876354/blog/1637819)
## **Inception V1**
通过设计一个稀疏网络结构，但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。谷歌提出了最原始Inception的基本结构：

![](\../img/note1/GoogleNetV1.png)

- 该结构将CNN中常用的卷积$（1\times1，3\times3，5\times5）$、池化操作$（3\times3）$堆叠在一起（卷积、池化后的尺寸相同，将通道相加），一方面增加了网络的宽度，另一方面也增加了网络对尺度的适应性。
- 网络卷积层中的网络能够提取输入的每一个细节信息，同时$5 \times 5$的滤波器也能够覆盖大部分接受层的的输入。还可以进行一个池化操作，以减少空间大小，降低过度拟合。在这些层之上，在每一个卷积层后都要做一个ReLU操作，以增加网络的非线性特征。
- 然而这个Inception原始版本，所有的卷积核都在上一层的所有输出上来做，而那个$5\times5$的卷积核所需的计算量就太大了，造成了特征图的厚度很大，为了避免这种情况，在$3\times3$前、$5\times5$前、max pooling后分别加上了1x1的卷积核，以起到了降低特征图厚度的作用，这也就形成了Inception v1的网络结构，如下图所示：

  ![](\../img/note1/googleNetV1_2.png)

  **$1\times1$卷积有什么用呢？**

  $1\times1$卷积的主要目的是为了减少维度，还用于修正线性激活（ReLU）。比如，上一层的输出为$100\times100\times128$，经过具有256个通道的5x5卷积层之后(stride=1，pad=2)，输出数据为$100\times100\times256$，其中，卷积层的参数为$128\times5\times5\times256= 819200$。而假如上一层输出先经过具有32个通道的$1\times1$卷积层，再经过具有256个输出的5x5卷积层，那么输出数据仍为为$100\times100\times256$，但卷积参数量已经减少为$128\times1\times1\times32 + 32\times5\times5\times256= 204800$，大约减少了**4倍**。

  **GooLeNet-V1网络结构**
  
  1. GoogLeNet采用了模块化的结构（Inception结构），方便增添和修改；
  2. 网络最后采用了average pooling（平均池化）来代替全连接层，该想法来自NIN（Network in Network），事实证明这样可以将准确率提高0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便对输出进行灵活调整；
  3. 虽然移除了全连接，但是网络中依然使用了Dropout ; 
  4. 为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度（**辅助分类器**）。辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中，这样相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。而在实际测试的时候，这两个额外的softmax会被去掉。

## **Inception V2**

**1.  卷积分解 factorizing convolutions**

大尺寸的卷积核可以带来更大的感受野，但也意味着会产生更多的参数，比如5x5卷积核的参数有25个，3x3卷积核的参数有9个，前者是后者的25/9=2.78倍。因此，GoogLeNet团队提出可以用2个连续的3x3卷积层组成的小网络来代替单个的5x5卷积层，即在保持感受野范围的同时又减少了参数量，如下图：

 ![](https://static.oschina.net/uploads/space/2018/0317/141655_QXsH_876354.png)

大卷积核完全可以由一系列的3x3卷积核来替代，那能不能再分解得更小一点呢？GoogLeNet团队考虑了nx1的卷积核，如下图所示，用3个3x1取代3x3卷积：

![](https://static.oschina.net/uploads/space/2018/0317/141700_GV5l_876354.png)

因此，<font color=red>**任意nxn的卷积都可以通过1xn卷积后接nx1卷积来替代**</font>。GoogLeNet团队发现在网络的前期使用这种分解效果并不好，**在中度大小的特征图（feature map）上使用效果才会更好（特征图大小建议在12到20之间）**。
![](https://static.oschina.net/uploads/space/2018/0317/141713_bGpL_876354.png)

**2. 降低特征图的大小**

一般情况下，如果想让图像缩小，可以有如下两种方式：

![](https://static.oschina.net/uploads/space/2018/0317/141726_LQNh_876354.png)

但是方法一（左图）先作pooling（池化）会导致特征表示遇到**瓶颈（特征缺失）**，方法二（右图）是正常的缩小，但计算量很大。为了同时保持特征表示且降低计算量，将网络结构改为下图，使用两个并行化的模块来降低计算量（卷积、池化并行执行，再进行合并）:

![](https://static.oschina.net/uploads/space/2018/0317/141734_OEPA_876354.png)

## **Inception V3**
Inception V3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算，又可以将1个卷积拆成2个卷积，使得网络深度进一步增加，增加了网络的非线性（每增加一层都要进行ReLU）。
另外，网络输入从224x224变为了299x299。

## **Inception V4**
Inception V4研究了**Inception模块与残差连接**的结合。ResNet结构大大地加深了网络深度，还极大地提升了训练速度，同时性能也有提升。

Inception V4主要利用残差连接（Residual Connection）来改进V3结构，得到Inception-ResNet-v1，Inception-ResNet-v2，Inception-v4网络。（结构图片见参考链接最后一部分哦~~）

![](https://static.oschina.net/uploads/space/2018/0317/141810_oD01_876354.png)

<br>

# **深度残差网络（DRN） ResNet**

[参考](https://my.oschina.net/u/876354/blog/1622896)

## **问题：反向传播过程中的梯度消失问题**

随着网络层级的不断增加，模型精度不断得到提升，而当网络层级增加到一定的数目以后，训练精度和测试精度迅速下降，这说明当网络变得很深以后，深度网络就变得更加难以训练了。

利用误差结果利用著名的<font color=red>**链式法则**</font>求偏导。通过不断迭代，对参数矩阵进行不断调整后，使得输出结果的误差值更小，使输出结果与事实更加接近。

从上面的过程可以看出，神经网络在反向传播过程中要不断地传播梯度，而当网络层数加深时，梯度在传播过程中会逐渐消失（假如采用Sigmoid函数，对于幅度为1的信号，每向后传递一层，梯度就衰减为原来的0.25，层数越多，衰减越厉害），导致无法对前面网络层的权重进行有效的调整。

那么，如何又能加深网络层数、又能解决梯度消失问题、又能提升模型精度呢？

<br>

## **深度残差网络（Deep Residual Network，简称DRN）**

假设现有一个比较浅的网络（Shallow Net）已达到了饱和的准确率，这时在它后面再加上几个恒等映射层（Identity mapping，也即y=x，输出等于输入），这样就增加了网络的深度，并且起码误差不会增加，也即更深的网络不应该带来训练集上误差的上升。而这里提到的**使用恒等映射直接将前一层输出传到后面的思想**，便是著名深度残差网络ResNet的灵感来源。

ResNet引入了**残差网络结构（residual network）**，通过这种残差网络结构，可以把网络层弄的很深（据说目前可以达到1000多层），并且最终的分类效果也非常好，残差网络的基本结构如下图所示，很明显，该图是带有跳跃结构的：

![](https://static.oschina.net/uploads/space/2018/0223/111635_C81Q_876354.png)

残差网络借鉴了**高速网络** （**Highway Network**）的跨层链接思想，但对其进行改进（残差项原本是带权值的，但ResNet用恒等映射代替之）。

假定某段神经网络的输入是x，期望输出是H(x)，即H(x)是期望的复杂潜在映射，如果是要学习这样的模型，则训练难度会比较大；

回想前面的假设，如果已经学习到较饱和的准确率（或者当发现下层的误差变大时），那么接下来的学习目标就转变为恒等映射的学习，也就是使输入x近似于输出H(x)，以保持在后面的层次中不会造成精度下降。

在上图的残差网络结构图中，通过“shortcut connections（捷径连接）”的方式，直接把输入x传到输出作为初始结果，输出结果为H(x)=F(x)+x，当F(x)=0时，那么H(x)=x，也就是上面所提到的恒等映射。于是，ResNet相当于将学习目标改变了，不再是学习一个完整的输出，而是目标值H(X)和x的差值，也就是所谓的残差F(x) := H(x)-x，因此，后面的训练目标就是要将残差结果逼近于0，使到随着网络加深，准确率不下降。

这种残差跳跃式的结构，打破了传统的神经网络n-1层的输出只能给n层作为输入的惯例，使某一层的输出可以直接跨过几层作为后面某一层的输入，其意义在于为叠加多层网络而使得整个学习模型的错误率不降反升的难题提供了新的方向。

说至此，神经网络的层数可以超越之前的约束，达到几十层、上百层甚至千层，为高级语义特征提取和分类提供了可行性。

![](https://static.oschina.net/uploads/space/2018/0223/111801_xFea_876354.png)

上图中的实线、虚线就是为了区分这两种情况的：

- <font color=red>实线的Connection部分</font>，表示通道相同，如上图的第一个粉色矩形和第三个粉色矩形，都是3x3x64的特征图，由于通道相同，所以采用计算方式为H(x)=F(x)+x
- <font color=red>虚线的的Connection部分</font>，表示通道不同，如上图的第一个绿色矩形和第三个绿色矩形，分别是3x3x64和3x3x128的特征图，通道不同，采用的计算方式为H(x)=F(x)+Wx，其中W是卷积操作，用来调整x维度的。

<br>

**三层残差学习单元：**

![](https://static.oschina.net/uploads/space/2018/0223/111833_m5OE_876354.png)

两种结构分别针对ResNet34（左图）和ResNet50/101/152（右图），其目的主要就是为了降低参数的数目。
- 左图是两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648
- 右图是第一个1x1的卷积把256维通道降到64维，然后在最后通过1x1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632
- 右图的参数数量比左图减少了16.94倍，因此，右图的主要目的就是<font color=red>为了减少参数量，从而减少计算量。</font>

对于常规的ResNet，可以用于34层或者更少的网络中（左图）；对于更深的网络（如101层），则使用右图，其目的是减少计算和参数量。

<br>

# **全文知识结构**

![](\../img/note1/hierarchical_structure.png)

在上面列举的经典的结构中，我们可以看出来网络越来越深。虽然越深的网络有更好的抽象表达能力和拟合能力，但是也会导致模型很难训练以及容易出现过拟合的。