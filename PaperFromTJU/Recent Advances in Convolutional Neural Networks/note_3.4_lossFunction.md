<!-- TOC -->

- [**Loss Function**](#loss-function)
  - [**Hinge Loss**](#hinge-loss)
  - [**Softmax Loss**](#softmax-loss)
  - [**Contrastive Loss(对比损失)**](#contrastive-loss%e5%af%b9%e6%af%94%e6%8d%9f%e5%a4%b1)
  - [**Triplet Loss**](#triplet-loss)
  - [**Kullback-Leibler Divergence（KL散度）**](#kullback-leibler-divergencekl%e6%95%a3%e5%ba%a6)

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

---
## **Softmax Loss**
softmax loss 常用于多分类问题，它将预测转。化为非负值，并将它们标准化得到对应所有类别的一个概率分布
> $p^{(i)}_j = e^{z^{(i)}_j} / \sum^K_{l=1} e^{z^{(i)}_l} \qquad$
> 
> $z^{(i)}_j通常是一个密集连接层的激活输出，可以写为z^{(i)}_j=w^T_ja^{(i)}+b_j$

> <font size=4px>$L_{softmax} = - \frac{1}{N}[\sum^N_{i=1}\sum^K_{j=1} 1\{{y^{(i)} = j}\} logp^{(i)}_j]$</font>

**Large-Margin Softmax (L-Softmax) loss**

4$在特征向量a^{(i)}和权重w_之间引入一个角度\theta_j，然后我们定义L-softmax的预测p^{(i)}_j为$
><font size=5px >$p^{(i)}_j = \frac{e^{\left \| w_j \right \|\left\|a^{(i)}\right\|} ψ(θ_j) }{e^{\left \| w_j \right \|\left\|a^{(i)}\right\|} ψ(θ_j) +  \sum_{l\neq j} e^{\left \| w_j \right \|\left\|a^{(i)}\right\|}\cos(\theta_l)}$ </font>
>
> <font size=4px>$ψ(θ_j ) = (-1)^k cos(mθ_j ) - 2k\quad, θ_j ∈ [kπ/m,(k + 1)π/m]$</font>

$k \in [0,m-1]为一个整数，m控制每个类型之间的间隔。$

当$m=1$时，$L-Softmax$变为普通的$softmax$。通过调整m的值，我们能得到一个能有效避免过拟合的比较复杂的学习对象


[L-Softmax loss 与 A-Softmax loss - 以下来源：知乎](https://www.zhihu.com/question/63247332/answer/222347446)

A-Softmax与L-Softmax的最大区别在于A-Softmax的**权重归一化了**，而L-Softmax则没的。A-Softmax权重的归一化导致特征上的点映射到单位超球面上，而L-Softmax则没有这个限制，这个特性使得两者在几何的解释上是不一样的。如图10所示，如果在训练时两个类别的特征输入在同一个区域时，如下图10所示。
![](https://pic2.zhimg.com/80/v2-a18d007d08f8cbd1ea7101a93aaeecae_hd.jpg)

图10：类别1与类别2映射到特征空间发生了区域的重叠

**A-Softmax只能从角度上分度这两个类别，也就是说它仅从方向上区分类**，分类的结果如图11所示；

![](https://pic4.zhimg.com/80/v2-43a87da5cc057414733ea5d120ab831a_hd.jpg)

图11：A-Softmax分类可能的结果

而**L-Softmax，不仅可以从角度上区别两个类，还能从权重的模（长度）上区别这两个类**，分类的结果如图12所示。
![](https://pic1.zhimg.com/80/v2-f583a7f8f705a35a7fd731f8e6559926_hd.jpg)

图12：L-Softmax分类可能的结果

在数据集合大小固定的条件下，**L-Softmax能有两个方法分类，训练可能没有使得它在角度与长度方向都分离，导致它的精确可能不如A-Softmax**。

-------

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
 > <font size=4px>$L_{d-contrastive} = \frac{1}{2N}\sum^N_{i=1}\sum^L_{l=1}(y)max(d_{(i,l)} - m_1, 0) + (1 - y)max(m_2 - d^{(i,l)}, 0)$</font>


----

## **Triplet Loss**

- Triplet loss最初是在 FaceNet: A Unified Embedding for Face Recognition and Clustering 论文中提出的，可以学到较好的人脸的embedding。用于训练**差异性较小**的样本，如人脸等。
- 通过优化锚（Anchor）和正样本距离小于锚与负样本距离实现样本相似性计算。

![](img\triplet_loss.png)
- **为什么不使用 softmax函数呢**
  - softmax最终的类别数是确定的，而Triplet loss学到的是一个好的embedding，相似的图像在embedding空间里是相近的，可以判断是否是同一个人脸。

这是一个含有三个实体的损失函数：$(x^{(i)}_a,x^{(i)}_p,x^{(i)}_n)$。通常包括anchor$x^{(i)}_a$以及和anchor同一个类别的正样本$x^{(i)}_p$和与anchor不同类别的负样本$x^{(i)}_n$。令$(z^{(i)}_a,z^{(i)}_p,z^{(i)}_n)$代表特征表示，loss如下：
> <font size=4px>$L_{triplet} =\frac{1}{N}\sum^N_{i=1}max{\{d^{(i)}_{(a,p)} - d^{(i)}_{(a,n)} + m, 0}\}$</font>

**[模型传送链接](https://www.jianshu.com/p/46c6f68264a1)**

>d表示特征向量之间的欧几里得距离。loss旨在最小化anchor和正样本的距离，最大化与负样本的距离。

但是随机选取anchor可能导致下面的情况: $d^{(i)}_{(n,p)} < d^{(i)}_{(a,p)} < d^{(i)}_{(a,n)}$，loss的结果依然是0，导致训练无效。下面提出**Coupled Cluster(CC)loss**来解决这个问题:
该方法是基于正、负样本集，将随机取样的anchor换成正样本的聚类中心，因此与负样本集合距离较远，避免了之前的问题。

> <font size=4px>$L_cc = \frac{1}{N^p} \sum^{N^p}_{i=1}\frac{1}{2} max\{\left \| z^{(i)}_p - c_p\right\|^2_2 -\left \| z^{(*)}_p - c_p\right\|^2_2+m,0\}$</font>

$N^P$是每组样本的数量,$z^{(*)}_p$是距离正样本的估计中点$c_p = (\sum^{N_p}_iz^{(i)}_p )/N^p$最近的负样本。

---

## **Kullback-Leibler Divergence（KL散度）**

KLD是在同一个离散变量x上两个不同的概率分布$p(x) 和 q(x)$的非对称评估（相似性）
![](\img/KLD.png)

从q(x)到p(x)的KLD定义为:
> $D_{KL}(p||q) = -H(p(x)) - E_p[log q(x)]$
> 
> <font size=4px>$\qquad \qquad \quad=\sum_x p(x)\log p(x) -\sum_xp(x) \log q(x) = \sum_xp(x) \log \frac{p(x)}{q(x)}$</font>
> 
> $H(p(x))是p(x)的信息熵，E_p[log q(x)]是p(x)与q(x)之间的交叉熵$

[以下参考](ttps://blog.csdn.net/matrix_space/article/details/80550561)

$D_{KL}(p||q)$ 表示的就是概率 q 与概率 p 之间的差异，很显然，散度越小，说明 概率 q 与概率 p 之间越接近，那么估计的概率分布于真实的概率分布也就越接近。

KL 散度可以帮助我们选择最优的参数，比如 $p(x)$ 是我们需要估计的一个未知的分布，我们无法直接得知 $p(x)$的分布，不过我们可以建立一个分布 $q(x|θ)$去估计 $p(x)$，为了确定参数 θ，虽然我们无法得知$p(x)$ 的真实分布，但可以利用采样的方法，从 $p(x)$ 中采样 N个样本，构建如下的目标函数：

> <font size=4px>$D_{KL}(p||q)=\sum^N_{i=1}{\{\log p(x_i)−\log q(x_i|θ)}\}$</font>

因为我们要预估的是参数 θ，上面的第一项 $\log p(x_i)$ 与参数 θ 无关，所以我们要优化的其实是 $−\log q(x_i|θ)$，而这个就是我们熟悉的最大似然估计。

- 【论文】

KL散度常用于AEs的变体：稀疏自动机、去噪自动机、Variational自动机（VAE）。VAE通过贝叶斯推理解释潜在的表示。自动机包括编码、解码两部分。编码把样本$x$转化为潜在表示$z \sim q_\phi(z|x)$，解码将潜在表示转化为模型的输入$x'\sim p_\theta(x|z)$。其中参数$\phi和\theta$是我们需要得到的参数。

VAEs利用最大化$\log p(x|\phi,\theta)$的对数似然下界：
> <font size=4px>$L_{vae}=E_{z\sim q_\phi (z|x)} [\log p_\theta(x|z)]  - D_{KL}(q_\phi(z|x)||p(z))$</font>

第一项是对输入样本重构的代价，对于KLD项强行设计$p(x)$是编码分布$q_\phi(z|x)$的先验。通常$p(z)$是标准正态分布、离散分布或某些具有几何解释的分布。

从原始VAE衍生出许多VAE的变体。条件VAE[75，84]用$x'∼p_θ(x|y，z)$从条件分布中生成样本。去噪VAE(DVEA)[74]从损坏的输入xˆ中恢复原始输入x。

下面提出**Jensen-Shannon Divergence(JSD)**，这是一种KLD的对称形式，它度量$p(x)和q(x)$之间的相似性：
> <font size=4px>$D_{JS}(p||q) = \frac{1}{2}D_{KL}(p(x)|| \frac{p(x)+q(x)}{2} )+ \frac{1}{2}D_{KL}(q(x)|| \frac{p(x)+q(x)}{2})$</font>

通过最小化JSD，我们可以使两个分布p(X)和q(X)尽可能接近。JSD已成功地应用于生成性对抗性网络(GANS)。相对于VAEs直接建模x和z之间关系的VAE，Gans被明确地设置为对生成任务进行优化。GANS的目标是寻找判别器D，它给出了真实数据和生成数据之间的最佳区分，同时鼓励生成器G对真实数据分布进行拟合。 判别器D与生成器G之间的最小-最大博弈由以下目标函数形式化：
> <font size=4px>$min_Gmax_DL_{gan}(D,G)=E_{x\sim p(x)}[ \log D(x)] + E_{z\sim q(z)}[\log (1-D(G(z)))]$</font>

相关论文证明了对于固定生成器$G^∗$，我们有一个最优判别器$D^∗_G(X)=\frac{ p(x)}{p(x)+q(x)}$.则上面的方程等价于最小化p(x)和p(x)之间的JSD。 

如果G和D具有足够的容量，则q(x)分布收敛于p(x)。与条件VAE一样，条件GaN(CGAN)也接收附加信息y作为输入以生成SAM。 在实践中，Gans训练的不稳定是出了名的。

- [以下参考](https://blog.csdn.net/FrankieHello/article/details/80614422?utm_source=copy)

**KL散度、JS散度和交叉熵**三者都是用来衡量两个概率分布之间的差异性的指标。不同之处在于它们的数学表达。

对于概率分布P(x)和Q(x)

**1. KL散度（Kullback–Leibler divergence）**

    又称KL距离，相对熵。当P(x)和Q(x)的相似度越高，KL散度越小。
    
    KL散度主要有两个性质：

   - 不对称性

    尽管KL散度从直观上是个度量或距离函数，但它并不是一个真正的度量或者距离，因为它不具有对称性，即D(P||Q)!=D(Q||P)。

   - 非负性

    相对熵的值是非负值，即D(P||Q)>0。

 

**2. JS散度（Jensen-Shannon divergence）**

JS散度也称JS距离，是KL散度的一种变形。
但是不同于KL主要又两方面：

（1）值域范围

    JS散度的值域范围是[0,1]，相同则是0，相反为1。相较于KL，对相似度的判别更确切了。

（2）对称性

    即 JS(P||Q)=JS(Q||P)，从数学表达式中就可以看出。

3）交叉熵（Cross Entropy）

    在神经网络中，交叉熵可以作为损失函数，因为它可以衡量P和Q的相似性。


**3. 交叉熵和相对熵的关系**：

在神经网络中，交叉熵可以作为损失函数，因为它可以衡量P和Q的相似性。

![](\img/交叉熵.jpg)

交叉熵和相对熵的关系：

![](\img/交叉熵和相对熵的关系.jpg)

以上都是基于离散分布的概率，如果是连续的数据，则需要对数据进行Probability Density Estimate来确定数据的概率分布，就不是求和而是通过求积分的形式进行计算了。