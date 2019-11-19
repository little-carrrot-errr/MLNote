<!-- TOC -->

- [**Upsampling上采样**](#upsampling%e4%b8%8a%e9%87%87%e6%a0%b7)
- [**Unpooling**](#unpooling)
- [**Deconvolution 反卷积**](#deconvolution-%e5%8f%8d%e5%8d%b7%e7%a7%af)
- [**Subsampled下采样**](#subsampled%e4%b8%8b%e9%87%87%e6%a0%b7)

<!-- /TOC -->

## **Upsampling上采样**

什么是上采样呢？简单来说：**上采样指的是任何可以让你的图像变成更高分辨率的技术**。
最简单的方式是**重采样**和**插值**：将输入图片进行rescale到一个想要的尺寸，而且计算每个点的像素点，使用如**双线性插值**等插值方法对其余点进行插值来完成上采样过程。

常见的上采样方法有 [以下参考](https://www.jianshu.com/p/587c3a45df67)：

**双线性插值**

双线性插值，又称为双线性内插。在数学上，双线性插值是对线性插值在二维直角网格上的扩展，用于对双变量函数（例如 x 和 y）进行插值。其核心思想是在两个方向分别进行一次线性插值。

假设我们想得到未知函数 f 在点 $P = (x, y)$ 的值，假设我们已知函数 f 在 $Q_{11} = (x1, y1)、Q_{12} = (x1, y2), Q_{21} = (x2, y1) 以及 Q_{22} = (x2, y2)$ 四个点的值。

首先在 x 方向进行线性插值，得到:![](\论文/img/note1/x.png)

然后在 y 方向进行线性插值，得到 f(x, y):![](\论文/img/note1/y.png)

在FCN中上采样用的就是双线性插值。

---

## **Unpooling**
**Unpooling**是在CNN中常用的来表示max pooling的逆操作。这是从2013年纽约大学Matthew D. Zeiler和Rob Fergus发表的《Visualizing and Understanding Convolutional Networks》中产生的idea：
> 鉴于max pooling不可逆，因此使用近似的方式来反转得到max pooling操作之前的原始情况

简单来说，记住做max pooling的时候的最大item的位置，比如一个3x3的矩阵，max pooling的size为2x2，stride为1，反卷积记住其位置，其余位置至为0就行:
![](\论文/img/note1/unpooling.png)

---

## **Deconvolution 反卷积**
**Deconvolution**(反卷积)在CNN中常用于表示一种反向卷积 ，但它并不是一个符合严格数学定义的反卷积操作。**与Unpooling不同**，使用反卷积来对图像进行上采样是可以<font color=red>**习得的**</font>。通常用来对卷积层的结果进行上采样，使其回到原始图片的分辨率。

反卷积也被称为**分数步长卷积**(convolution with fractional strides)**或者转置卷积**(transpose convolution)或者**后向卷积**(backwards strided convolution)。作者认为应该称之为**转置卷积**。

常规的卷积操作(valid模式)：滑动步长为S，图片大小为N1xN1，卷积核大小为N2xN2（示意图假设为3x3），卷积后图像大小：$(N1-N2)/S+1 \times (N1-N2)/S+1$如下图：

![](\论文/img/note1/convolution.gif)

为了要让经过卷积的结果回到卷积前的模样。如图这个2x2的的结果，如何回到4x4呢？其实这种也算一种卷积的操作，只不过进行了padding操作：

![](\论文/img/note1/deconvolution.gif)

或者，通常大家会选择更大的dilation（等价于小于1的stride）来增强上采样的效果（可以理解成分数步长卷积，下图显示的是stride=1/2的示意图）:

![](\论文/img/note1/deconvolution_with_0.5stride.gif)


---
## **Subsampled下采样**

缩小图像（或称为下采样（subsampled）或降采样（downsampled））的主要目的有两个：1、使得图像符合显示区域的大小；2、生成对应图像的缩略图

下采样原理：对于一副图像I尺寸为$M*N$，对起进行s倍下采样，即得到$（M/s）*（N/s）$尺寸的分辨率图像，当然，s应该是M和N的公约数才可以，如果考虑是矩阵形式的图像，就是把原始图像s*s窗口内的图像变成一个像素，这个像素点的值就是窗口内所有像素的均值。<font size=4.5px>$P_k = \sum \frac{I_i}{s^2}$</font>