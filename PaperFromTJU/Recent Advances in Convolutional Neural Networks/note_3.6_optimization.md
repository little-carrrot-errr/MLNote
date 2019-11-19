<!-- TOC -->

- [**Optimizaition**](#optimizaition)
  - [**Data Augmentation 数据增强**](#data-augmentation-%e6%95%b0%e6%8d%ae%e5%a2%9e%e5%bc%ba)
  - [**Weight Initialization 权重初始化**](#weight-initialization-%e6%9d%83%e9%87%8d%e5%88%9d%e5%a7%8b%e5%8c%96)
  - [**Stochastic Gradient Descent 随机梯度下降**](#stochastic-gradient-descent-%e9%9a%8f%e6%9c%ba%e6%a2%af%e5%ba%a6%e4%b8%8b%e9%99%8d)
  - [**Batch Normalization**](#batch-normalization)
  - [**Shortcut Connection**](#shortcut-connection)

<!-- /TOC -->

# **Optimizaition**

## **Data Augmentation 数据增强**
深度CNN十分依赖于大量可用的训练数据，因此比起CNN中大量的参数，减轻数据的相对稀缺性的一个优雅的办法是**数据增强**。数据增强对可用数据进行变化，在不改变原数据的特性上得到新的可用数据。常用的数据增强方式包括简单的**几何变化，如：采样、镜像、旋转、位移、各种光度转换。**[97]提出一种通过贪心策略在一群候选变化中选择最佳变化但是这种策略包括大量模型预训练步骤，当预选变化数量很多的时候计算量很大。[98]提出了一种通过随机生成**不同同胚(diffeomorphisms)** 来增强数据的优雅方法。[100]提供了从互联网收集图像的额外手段，以改进细粒度识别任务中的学习。

---
## **Weight Initialization 权重初始化**
深度CNN有大量的参数，其损失函数是非凸的[101]，这给训练带来了很大的困难。为了在训练中实现快速收敛，避免消失梯度问题，适当的网络初始化是最重要的先决条件之一。偏置项参数可初始化为零，而权重参数应被谨慎初始化，**并且须要打破同一层隐单元间的对称性（the symmetry among hidden units of the same layer）**。

下面举个栗子：
假设每层线性放大输入$k$倍，那么对于$L$层神经网络将放大输入$k^L$倍。在这个情况下如果$k > 1$，那么我们输出层就是一个很大的值，如果$k<1$那么输出就是一个小到diminishing的值和梯度。

【**改进**】
- Krizhevsky等人[8]用标准偏差0.01的零均值高斯分布初始化其网络的权值，并设置第二、第四和第五卷积项的偏置项以及所有全连接层都是常量**1**。
- 另一个著名的随机初始化方法是“Xavier”，这是在[104]中提出的。它们从均值为零且方差为$2/(n_{in} + n_{out})$的高斯分布中选择权重，其中$n_{in}$为输入的神经元数，$n_{out}$是输入结果的神经元数。因此Xavier可以根据输入输出的神经元数量动态初始化参数，并且保持多个神经层的signal在一个合理的范围。
  - 它的一个变体[Caffe](https://github.com/BVLC/caffe)使用了只使用$n_{in}$的变体，这使得它更易于实现。
- “Xavier”初始化方法后来被扩展为[56]，以解释校正的非线性。其中，它们特别考虑了RELU非线性导出了一种鲁棒的初始化方法。这个方法能使得很深的模型训练到收敛，而Xavier不行。

【题外】

【方差校准】
**经验告诉我们如果初始时每个单元的输出都有着相似的分布会使收敛速度增大。使用随机的方式会使得各个单元的输出值的分布产生较大的变化，先假设使用线性激活函数，探究输入与输出的分布的关系。**[参考](https://blog.csdn.net/bea_tree/article/details/51519844)

[105]表明，正交矩阵初始化对于线性网络比高斯初始化要好得多，对于非线性的网络也是如此。[102]将[105]扩展到迭代过程。具体的，它提出一种层序单元方差过程模式，这个模式可以看出正交初始化结合批归一（batch normalization）并只使用first mini-batch。它类似于批归一化，因为两者都采用单位方差归一化过程。不同的是，它使用正交归一化来初始化权重，这有助于有效地实现de-correlate(去相关) layer activities。这种初始化技术已应用于[106，107]，使得性能显著提高。

---
## **Stochastic Gradient Descent 随机梯度下降**
标准的梯度下降更新参数$\theta$的公式为：$\theta_{t+1}=\theta_t -\eta \nabla_\theta E[L(\theta_t)]$。其中$E[L(\theta_t)]$是$L(\theta)$在所有训练集上的期望。$\eta$是学习率。

与$E[L(\theta_t)]$不同，SGD基于训练集的单个随机选取的样本$(x^{(t)},y^{(t)})$来估计偏差：
> <font size=4px>$\theta_{t+1}=\theta_t -\eta_t \nabla_\theta L(\theta_t;x^{(t)},y^{(t)})$</font>

在实际训练中，每次SGD的参数训练都是使用mini-batch而不是在单个的样本上进行的。这种方式可以帮助减 少参数更新过程的方差，并且达到更加稳定的收敛。但是SGD还是有一些问题需要克服：
1. 选择一个合适的$\eta_t$是比较困难的。一种常用的方法是使用恒定的学习速率，在初始阶段给出稳定的收敛速度，然后随着收敛速度的减慢而降低学习速度。此外，提出了学习率附表[110，111]来调整训练期间的学习率。
2. 为了使当前梯度更新依赖于历史批次和加速训练，提出了动量 **momentum** [108]在相关方向累积速度矢量。**经典momentum最新情况如下：**
    > <font size=4px>$v_{t+1} = \gamma v_t - \eta_t \nabla_θ L(θ_t;x^{(t)},y^{(t)})$
    >  
    > $θ_{t+1} = θ_t + v_{t+1}$</font>

    $\gamma常设置为0.9 \qquad v_{t+1}是当前动量值$
- **Nesterov momentum**是另一种使用动量梯度下降的方法：
    > <font size=4px>$v_{t+1} = \gamma v_t - \eta_t \nabla_θ L(θ_t+\gamma v_t;x^{(t)},y^{(t)})$</font>

    与传统方法比较，传统方法先计算出当前的梯度，然后朝着梯度的方向更新累积动量；而Nesterov动量首先向上一次累积梯度$\gamma v_t$方向移动，然后再计算梯度并更新。这种预期的更新避免了优化速度过快，并获得了更好的性能。
- **Parallelized SGD**优化了SGD，它适应于大型、并行的机器学习。
    > 与标准(同步)SGD不同的是，如果其中一台机器速度慢，训练将被延迟。这些并行化方法使用异步机制，这样就不会影响有其他优化。 除了那台慢的机器上延迟了。
- JeffreyDean等人[114]使用另一种异步的SGD过程称为**Downpour SGD**，以加快多CPU集群上的大规模分布式培训过程。
- Paine等人[115]基本上将异步SGD和GPU结合起来，使训练时间比靠一台机器快几倍。 
- 庄等人[116]还使用多个GPU异步计算梯度和更新全局模型参数，在4个GPU上实现了相比一个GPU上训练的3.2倍的加速

【提早停止方法】
SGD方法可能不会导致收敛。当性能停止改善时，可以终止训练过程。过度训练的一个流行的补救方法是使用早期停止[117]，在这种方法中，可以基于训练期间验证集的性能来停止优化。为了控制训练过程的次数，可以考虑各种停止标准。例如，可以使用固定迭代次数来执行训练，或直到达到预定义的training error[118]为止。在提高网络泛化能力和避免过拟合的前提下，适当的停止策略应该让训练过程继续进行。

---
## **Batch Normalization**

数据规范化通常是数据预处理的第一步。全局数据归一化将所有数据转换为零均值和单位方差.然而，当数据通过一个很深的NNetwork, 输入到内部层的分布将发生变化，这将降低网络的学习能力和准确性。

[120]提出一个有效的方法：**Batch Normalization(BN)**。

它通过一个归一化步骤来完成所谓**的协变量移位问题（covariate shift problem）**，该步骤确定了层输入的均值和方差，其中平均和方差的估计是在每次mini-batch之后计算出来的，而不是计算整个训练集。

假设要规范化的层有一个d维输入，$x=[x_1,x_2,...,x_d]^T$，我们首先正则化第k维度如下：
> <font size = 4px>$x'_k = (x_k - \mu_\beta) / \sqrt{\sigma^2_\beta + \epsilon}$</font>

>$\mu_\beta为mini-batch的期望，\sigma^2_\beta为方差，\epsilon为一个常量$

为了加强表达能力，将正则化输入$x'_k$转化为：
> <font size=4px>$y_k =BN_{\gamma,\beta}(x_k)=\gamma x'_k + \beta$</font>

其中γ和β是需要学习的参数。Batch Normalization相比全局数据正则化有很多优点。
1. 他减少了**内部存在协方差偏移（Internal Covariate Shift）现象**：深度网络内部数据分布在训练过程中发生变化的现象
2. BN降低了梯度对参数的尺度或初值的依赖，从而对网络中的梯度流产生了有利的影响(**使得梯度变得平)缓**。这样就可以使用较高的学习率，而不会有发散的危险。
3. BN规范了模型，从而减少了对Dropout的需求。(**使模型正则化具有正则化效果**)
4. bn使得在不陷入饱和模型的情况下使用饱和非线性激活函数成为可能。(**优化激活函数**)

[其他参考](https://www.jianshu.com/p/a78470f521dd)

---

## **Shortcut Connection**

如上所述，通过归一初始化[8]和BN[120]，可以缓解深层CNN的消失梯度问题。这些方法虽然成功地防止了深层神经网络的过度拟合，但也给网络的优化带来了困难，造成了比浅层神经网络更差的性能。 更深层次的CNN所遭受的这种优化问题被认为是退化问题(**degradation problem**)。

**LSTM \ Long Short Term Memory**使用gate function来决定一个神经元的激活值有多少比重呗转化还是只是pass through。受这个启发，Srivastava等人[122]提出能够优化几乎任意深度网络的**Highway networks**。网络结构如下：
> <font size=4px>$x_{l+1} = \phi_{l+1}(x_l,W_H)\cdot \tau_{l+1}(x_l.W_T)+ x_l\cdot(1-\tau_{l+1}(x_l,W_T))$</font>

> $x_l和x_{l+1}代表第l个highway 块的输入和输出。\tau(\cdot)是装换门，\phi(\cdot)通常是$一个仿射变换【指在几何中，一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间】 跟随着一个非线性激活函数（可以是其他形式）。
 
这种选通机制迫使该层的输入和输出具有相同的大小，并允许对具有数十层或数百层的公路网进行有效的培训。gate的输出随输入示例的不同而有很大差异，这表明网络不仅学习固定的结构，而且根据特定的样例动态地路由数据。**（dynamically routes data based on specific examples）**

[下面是highway network详解](https://blog.csdn.net/qq_27009517/article/details/84028568)：
![](/img/highway&#32;network.png)

残差网络(ResNets)[12]有着与LSTM单元相同的核心思想。与通过学习神经元特定门控的参数不同，ResNet中的快捷连接没有gated也没有transformed输入，而是直接传播到输出，这会带来更少的参数：
> <font  size = 4px>$x_{l+1} = x_l + f_{l+1}(x_l,W_F)$</font>
> 
> $f_l 是一个权重层，他可以是全连接，BN，ReLU或者Pooling的组合。$

通过残差块，任何深层的单元都可以被写成一个稍浅层单元和一个残差函数之和。这也意味着梯度可以直接传播到更浅的单元，这使得深层的网络相 比前面的映射函数的形式更容易被优化，并且更有效地进行训练。这与通常的前馈网络不同，通常梯度本质上是一系列矩阵向量乘积，随着网络的深入，这些乘积可能会消失。

- 在原始ResNet之后，He等人[123]发布了另一个重新激活的Resnets变体，在那里他们执行一组实验以显示**identity shortcut connections** 是最易于供网络学习的。他们还发现，将BN提前执行比之后添加BN效果要好得多。在他们的比较中，和之前的以前的ResNets[12]相比，BN+ReLU预激活的残差网络的准确性有很大的提高。
- 受[123]的启发，Shen等[124]，[124]引入了从卷积层输出的加权因子，该加权因子逐渐引入可训练层。
- 最新的“Inception-v4”论文[42]还报告说，在inception module中使用**identity skip connections**，可以加训练并提高性能。
- 原始的Resnet和预激活Resnet是非常深的，但也非常薄。相反，**Wide Resnet**[125]提出减小深度和增加宽度，对CIFAR-10，CIFAR-100和SVHN达到了很不错的效果。然而，在Imagenet数据集上的大规模图像分类任务上，它们的声明没有得到验证。
- 随机深度ResNets（**Stochastic Depth ResNets**）随机丢弃一个层的子集，并使用 **identity mapping**在mini-batch 绕过它们。通过结合随机深度Resnet和Dropout，Singh等。[126]提出了随机深度的dropout和网络，可以看作是Resnet、Dropout和Stochastic Depth ResNets的集合
- **The Resnet in ResNets\RIR**论文[127]中的描述了一种将经典卷积网络和残差网络结合在一起的体系结构，其中每个RIR块包含残差单元和非残差块。RIR可以学习每个残差块应该使用多少卷积层。
- **ResNets of ResNets\RoR**[128]是对ResNets体系结构的修改，它建议使用多级shortcut connections，而不是先前工作中的单层connection
-  **DenseNet**[129]可以被看作是一种将skip connection的洞察力带到了极致的架构。结构中每层的输出连接到其所有后续层

在所有的ResNets[12，123]，Highway[122]和inception network[42]，我们可以看到一个相当明显的趋势：使用skip connection以帮助训练非常深的网络。