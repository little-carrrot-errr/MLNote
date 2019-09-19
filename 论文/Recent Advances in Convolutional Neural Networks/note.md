<!-- TOC -->

- [**Loss Function**](#loss-function)
  - [**Hinge Loss**](#hinge-loss)
  - [**Softmax Loss**](#softmax-loss)
  - [**Contrastive Loss(对比损失)**](#contrastive-loss%e5%af%b9%e6%af%94%e6%8d%9f%e5%a4%b1)

<!-- /TOC -->
# **Loss Function**
## **Hinge Loss**
在机器学习中，hinge loss作为一个损失函数(loss function)，通常被用于**最大间隔算法(maximum-margin)**，而最大间隔算法又是**SVM(支持向量机support vector machines)用到的重要算法**(注意：SVM的学习算法有两种解释：1. 间隔最大化与拉格朗日对偶；2. Hinge Loss)。

Hinge loss专用于二分类问题，标签值y=±1，预测值y^∈R。该二分类问题的目标函数的要求如下：
  当y^大于等于+1或者小于等于-1时，都是分类器确定的分类结果，此时的损失函数loss为0；而当预测值y^∈(−1,1)时，分类器对分类结果不确定，loss不为0。显然，当y^=0时，loss达到最大值。

> 对于输出y=±1，当前y^的损失为：$ℓ(y)=max(0,1−y⋅y')$

上式是Hinge loss在二分类问题的变体[参考](https://blog.csdn.net/hustqb/article/details/78347713)

> 对于多分类$y^{(i)}\in [1,...,K]$ ：$L_{hinge} =\frac{1}{N} \sum^N_{i=1}\sum^N_{j=1}[\max(0,1-\sigma(y^{(i)},j)w^Tx_i)]^p$

$当y^{(i)}=j时，\sigma(y^{(i)},j)=1；否则\sigma(y^{(i)},j)=-1$。

当p=1时，上式为$Hinge-Loss(L_1 - Loss)$，当p=2时，上式为$Squared Hinge-Loss(L_2 - Loss)$

## **Softmax Loss**
softmax loss 常用于多分类问题，它将预测转。化为非负值，并将它们标准化得到对应所有类别的一个概率分布
> $p^{(i)}_j = e^{z^{(i)}_j} / \sum^K_{l=1} e^{z^{(i)}_l} \qquad$
> 
> $z^{(i)}_j通常是一个密集连接层的激活输出，可以写为z^{(i)}_j=w^T_ja^{(i)}+b_j$

> $L_{softmax} = - \frac{1}{N}[\sum^N_{i=1}\sum^K_{j=1} 1\{{y^{(i)} = j}\} logp^{(i)}_j]$

**Large-Margin Softmax (L-Softmax) loss**

$在特征向量a^{(i)}和权重w_之间引入一个角度\theta_j，然后我们定义L-softmax的预测p^{(i)}_j为$
><font size=5px >$p^{(i)}_j = \frac{e^{\left \| w_j \right \|\left\|a^{(i)}\right\|} ψ(θ_j) }{e^{\left \| w_j \right \|\left\|a^{(i)}\right\|} ψ(θ_j) +  \sum_{l\neq j} e^{\left \| w_j \right \|\left\|a^{(i)}\right\|}\cos(\theta_l)}$ </font>
>
> <font size=4px>$ψ(θ_j ) = (-1)^k cos(mθ_j ) - 2k\quad, θ_j ∈ [kπ/m,(k + 1)π/m]$</font>

$k \in [0,m-1]为一个整数，m控制每个类型之间的间隔。$

当$m=1$时，$L-Softmax$变为普通的$softmax$。通过调整m的值，我们能得到一个能有效避免过拟合的比较复杂的学习对象


[L-Softmax loss 与 A-Softmax loss - 以下来源：知乎](https://www.zhihu.com/question/63247332/answer/222347446)

A-Softmax与L-Softmax的最大区别在于A-Softmax的**权重归一化了**，而L-Softmax则没的。A-Softmax权重的归一化导致特征上的点映射到单位超球面上，而L-Softmax则不没有这个限制，这个特性使得两者在几何的解释上是不一样的。如图10所示，如果在训练时两个类别的特征输入在同一个区域时，如下图10所示。
![](https://pic2.zhimg.com/80/v2-a18d007d08f8cbd1ea7101a93aaeecae_hd.jpg)

图10：类别1与类别2映射到特征空间发生了区域的重叠

**A-Softmax只能从角度上分度这两个类别，也就是说它仅从方向上区分类**，分类的结果如图11所示；
![](https://pic4.zhimg.com/80/v2-43a87da5cc057414733ea5d120ab831a_hd.jpg)

图11：A-Softmax分类可能的结果

而**L-Softmax，不仅可以从角度上区别两个类，还能从权重的模（长度）上区别这两个类**，分类的结果如图12所示。
![](https://pic1.zhimg.com/80/v2-f583a7f8f705a35a7fd731f8e6559926_hd.jpg)

图12：L-Softmax分类可能的结果

在数据集合大小固定的条件下，**L-Softmax能有两个方法分类，训练可能没有使得它在角度与长度方向都分离，导致它的精确可能不如A-Softmax**。

## **Contrastive Loss(对比损失)**

Contrastive Loss常用于训练Siames network（孪生神经网络，一种弱监督学习用于从标记为匹配或非匹配的数据实例对中学习相似性度量）

$对于给定第i组输入数据： (x^{(i)}_\alpha,x^{(i)}_\beta)，令(z^{(i,l)}_\alpha,z^{(i,l)}_\beta)为第l层的对应输出对 （l\in [1,...,L]）。$

在相关研究中，将输入图像通过两个独立的CNNs，然后将最后一层的特征向量输入到成本模块，他们使用$Contrastive function来作为损失函数训练样本：$

<font size=4px>$L_{contrastive} = \frac{1}{2N} \sum^N_{i=1}(y)d^{(i,L)} + (1-y)max(m-d^{(i,L)},0)$
<br>
<br>
$d^{(i,L)} = ||z^{(i,L)}_α - z^{(i,L)}_β ||^2_2 代表了两个样本的欧氏距离$
<br>
</font>

y为两个样本是否匹配的标签，y=1代表两个样本相似或者匹配，y=0则代表不匹配，m为设定的阈值

观察上述的contrastive loss的表达式可以发现，这种损失函数可以很好的表达成对样本的匹配程度，也能够很好用于训练提取特征的模型。
- 当y=1（即样本相似）时，损失函数只剩下$\frac{1}{2N} \sum^N_{i=1}(y)d^{(i,L)}$，即原本相似的样本，如果在特征空间的欧式距离较大，则损失函数（增函数）越大，则说明当前的模型不好
- 而当y=0时（即样本不相似）时，损失函数为$(1-y)max(m-d^{(i,L)},0)$（减函数），即当样本不相似时，其特征空间的欧式距离反而小的话，损失函数值会变大。
![](https://img-blog.csdn.net/20161113163224993)

这张图表示的就是损失函数值与样本特征的欧式距离之间的关系，其中**红色虚线表示的是相似样本的损失值（Y=1时），蓝色实线表示的不相似样本的损失值（Y=0时）**。这里的m为阈值，要视具体问题而定，对于不同的目标m的值会有不同的大小。而事实表明Constractive Loss对于多分类的问题经常会在训练集上过拟合，显得比较乏力。针对该问题的改进方法有Triplet Loss、四元组损失(Quadruplet loss)、难样本采样三元组损失(Triplet loss with batch hard mining, TriHard loss)、边界挖掘损失(Margin sample mining loss, MSML)
[原文链接](https://blog.csdn.net/qq_37053885/article/details/79325892)

在训练中，发现loss对于不匹配的稳定性高，而对于匹配部分，微调的参数会导致性能下降。因此为了解决这个问题，他们在匹配部分也加入一个间隔参数，在每一层神经层都添加了一个contrastive loass以及它对应的backpropagation：
><font size=5px>$L_{d-contrastive} = \frac{1}{2N}\sum^N_{i=1}\sum^L_{l=1}(y)max(d_{(i,l)} - m_1, 0) + (1 - y)max(m_2 - d^{(i,l)}, 0)$</font>