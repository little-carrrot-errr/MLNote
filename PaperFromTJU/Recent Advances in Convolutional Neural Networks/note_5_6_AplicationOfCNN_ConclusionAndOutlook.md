<!-- TOC -->

- [**Application of CNNs**](#application-of-cnns)
  - [**Image Classification图像分类**](#image-classification%e5%9b%be%e5%83%8f%e5%88%86%e7%b1%bb)
  - [**Object Tracking(对象跟踪)**](#object-tracking%e5%af%b9%e8%b1%a1%e8%b7%9f%e8%b8%aa)
  - [**Pose Estimation姿态估计**](#pose-estimation%e5%a7%bf%e6%80%81%e4%bc%b0%e8%ae%a1)
  - [**Test Detection and Recognition 文本检测和识别**](#test-detection-and-recognition-%e6%96%87%e6%9c%ac%e6%a3%80%e6%b5%8b%e5%92%8c%e8%af%86%e5%88%ab)
    - [**Text Detection 文本检测**](#text-detection-%e6%96%87%e6%9c%ac%e6%a3%80%e6%b5%8b)
    - [**Text Recognition文本识别**](#text-recognition%e6%96%87%e6%9c%ac%e8%af%86%e5%88%ab)
    - [**条件随机场**](#%e6%9d%a1%e4%bb%b6%e9%9a%8f%e6%9c%ba%e5%9c%ba)
    - [**自然语言处理中的N-Gram模型详解**](#%e8%87%aa%e7%84%b6%e8%af%ad%e8%a8%80%e5%a4%84%e7%90%86%e4%b8%ad%e7%9a%84n-gram%e6%a8%a1%e5%9e%8b%e8%af%a6%e8%a7%a3)
    - [**End-to-end Text Spotting端到端文本检测**](#end-to-end-text-spotting%e7%ab%af%e5%88%b0%e7%ab%af%e6%96%87%e6%9c%ac%e6%a3%80%e6%b5%8b)
  - [**Visual Saliency Detection视觉显著性检测**](#visual-saliency-detection%e8%a7%86%e8%a7%89%e6%98%be%e8%91%97%e6%80%a7%e6%a3%80%e6%b5%8b)
  - [**Action Recognition行为识别**](#action-recognition%e8%a1%8c%e4%b8%ba%e8%af%86%e5%88%ab)
    - [**Action Recognition in Still Images**](#action-recognition-in-still-images)
    - [**Action Recognition in Video Sequences在视频序列中的动作识别**](#action-recognition-in-video-sequences%e5%9c%a8%e8%a7%86%e9%a2%91%e5%ba%8f%e5%88%97%e4%b8%ad%e7%9a%84%e5%8a%a8%e4%bd%9c%e8%af%86%e5%88%ab)
  - [**Scene Labeling场景标记**](#scene-labeling%e5%9c%ba%e6%99%af%e6%a0%87%e8%ae%b0)
  - [**Speech Processings**](#speech-processings)
    - [**Automatic Speech Recognition**](#automatic-speech-recognition)
    - [**Statistical Parametric Speech Synthesis统计参数语音合成**](#statistical-parametric-speech-synthesis%e7%bb%9f%e8%ae%a1%e5%8f%82%e6%95%b0%e8%af%ad%e9%9f%b3%e5%90%88%e6%88%90)
  - [**Natural Language Processing自然语言处理**](#natural-language-processing%e8%87%aa%e7%84%b6%e8%af%ad%e8%a8%80%e5%a4%84%e7%90%86)
    - [**Statistical Language Modeling**](#statistical-language-modeling)
    - [**Text Classification文本分类**](#text-classification%e6%96%87%e6%9c%ac%e5%88%86%e7%b1%bb)
- [**Conclusion and Outlook**](#conclusion-and-outlook)

<!-- /TOC -->

# **Application of CNNs**

## **Image Classification图像分类**
CNNs在图像分类中的应用已经很长时间了[168-171]。与其他方法相比，由于CNN在组合特征和分类学习中的能力，它可以在大规模数据集上获得更好的准确性[8,9,172]。大规模图像分类的突破出现在2012年。Krizhevsky等人[8]开发AlexNet，并在2012年ILSVRC中取得最佳成绩。在AlexNet成功之后，许多方法通过缩小过滤器大小[11]或扩大网络深度[9，10]使得CNN在分类精度上取得了显著的提高。

在多标签分类中，构建一个结构化的分类器是一个常用的图像分类的方法[173]。
- [174]的工作是在cnn中引入类别层次的最早尝试之一，其中提出了一种基于树优先（**tree-based**）的区分转移学习方法（**discriminative transfer learning**）。他们使用一种类的层次结构在相关类之间共享信息，以提高训练样例少的类别的性能。
- Wang等人[175]建立一个树结构来学习细粒度特征用于子类别识别
- Xiao等人[176]提出了一种训练方法，该方法不仅可以逐步增长网络，而且可以分层增长网络。首先使用粗分类CNN分类器将容易区分的类彼此分离，然后将那些难以分类的类路由到下游的精细类别分类器，以便进一步的预测。该体系结构遵循粗糙到精细的分类模式，可以以可承受的复杂度增加为代价，获得较低的误差。

子类别（子标签）分类是另一类发展迅速的图像识别的领域。已经有一些细粒度的图像数据集(如鸟类[178]、狗[179]、汽车[180]和植物[181])。使用对象部分信息（object part information）有利于细粒度分类[182]。一般情况下，通过定位物体的重要部分并以区分度大的方式来表示它们的外形，可以训练的提高精度。
- Branson等[183]提出了一种检测局部的方法，并从多姿态归一化区域（**multiple pose-normalized regions**）提取CNN特征。他们还建立了一个模型，将低层特征层与姿态规范化提取例程相结合，并将更高级别的特征层与**未对齐图像**([image alignment](https://blog.csdn.net/h763247747/article/details/100862863))特征结合起来，以改进分类的准确性。
- Zhang等人【184】提出了一种基于局部的R-CNN方法，该方法可以学习全目标和局部检测器。他们使用 **selective search**（[参考](https://www.cnblogs.com/zyly/p/9259392.html)）来生成局部区域（**part proposal**），并使用非参数几何约束 **non-parametric geometric constraints** (采用了几何约束，将块检测的结果约束在物体检测的一定范围内)来获得更加准确地局部定位。[参考](https://blog.csdn.net/sheng_ai/article/details/41806341)
- Lin等人[186]将局部定位、图像对齐以及图像分类组合到一个识别系统中，其名为 **Deep LAC**。系统由**三个子网组成**：
  - 定位子网用于估计局部位置，输出为框的左上角及右下角点的坐标
  - 对齐子网络接收部件定位结果，执行模板对齐，产生姿态对齐的部件图像 [187],对齐子网络进行平移、缩放、旋转等操作用于姿态对齐区域的生成。同时，该子网络还负责反向传播过程中分类及定位结果的桥接作用。
  - 分类子网络以位姿对齐部分图像作为输入来预测类别标签。
- 他们还提出了一个**阀门连接函数 value linkage function**来连接子网络，并使它们在训练和测试中作为一个整体。

需要注意的是，上述所有方法都是利用部分注释信息进行监督培训的。然而，这些注释并不容易收集，而且这些系统很难扩展和以及难以处理许多类型的细粒度类。为了避免这一问题，一些研究人员开始研究在没有监督的情况下找到局部的部分或区域的问题。

[见微知著——细粒度图像分析进展综述 ](https://www.sohu.com/a/134764420_473283)
- Krause等人[188]将局部学习特征表示的集合用于细粒分类，他们使用**共同分割和对齐** **co-segmentation and alignment** 来生成局部，然后比较各个部分的不同，合并类似的局域。在他们最新的论文中【189】，他们将共分割和对齐结合在一个区分性的混合模型（**discriminative mixture**）中，以生成有利于细粒度分类的部件。
- Zhang等人[190]使用**无监督的选择性搜索生成目标proposal**，然后从多尺度生成的局部proposals中选择有用的部分
- Xiao等人[191]在cnn中应用视觉注意机制（**visual attention**）进行细粒度分类。他们的分类器主要分为是哪个阶段：1. bottom-up attention：获取多个候选框；2. object-level top-down机制选择某个物体的相关的patches；3. 在part level top-down机制中定位区分不同的部分：
![](/img/Two&#32;Level&#32;Attention&#32;Model.png)

   - 将这三个结构结合进行训练得到的网络对于查找前景对象或对象部分，并提取识别特征有很大帮助。
 - 林等人[192]提出了一种用于**细粒度图像分类的双线性模型**。识别结构由两个特征提取器组成。两个特征提取器的输出在图像的每个位置使用外部乘积进行相乘，并汇集以获得 **an image descriptor**

---
## **Object Tracking(对象跟踪)**
对象跟踪的成功很大程度上依赖于目标外观的表示对几个挑战（如视点变化  **view point changes**、光照变化 **illumination changes** 和闭塞**occlusions**）的鲁棒性如何[213] –215]。有许多尝试将CNN应用于对象追踪中。
- Fan等人【216】将CNN作为基础学习器。它学习一个单独的特定于类的网络来跟踪对象。在【216】中，作者利用**位移偏差结构shift-variant architecture**设计了一个CNN tracker/CNN跟踪器。这种架构起着关键的作用，它将CNN模型从检测器转变为跟踪器。这些功能是在离线培训期间学习的。 与仅提取局部空间结构的传统跟踪器不同，此基于CNN的跟踪方法通过考虑两个连续帧的图像来提取空间和时间结构。由于时间信息中的大信号倾向于出现在正在移动的物体附近，因此时间结构为跟踪提供了原始速度信号
- j- Li等人【217】提出了一个目标特定的cnn用于目标跟踪，其中cnn在跟踪过程中通过在线获得的新示例进行增量训练。它们使用多个CNNs的候选池作为目标对象的不同实例的数据驱动模型。个体上，每个CNN维护一组特定的kernels，可以利用低级的线索（low-level cues）很好的将物体从他们周围环境中区分出来。这些核在对应的CNN初始化时仅用一个实体训练，之后这些核将会用过在线的方式在每一帧被更新。
  
    与过去对所有的对象训练一个复杂而有效的CNN模型不同，他们在具有实时更新机制的框架内，在CNN中使用一些相对较少的filters。给定一帧，池中最好的CNN将会被选择出来作为判别对象假设。最高分的假设将会被设置为当前物体的检测窗口，被选择的模型使用热启动反向传播重新训练，从而优化结构损失函数。
-  在[218]中，提出了一种cnn目标跟踪方法，以解决目标跟踪问题中手工特征和浅分类器结构的局限性。识别特征首先通过CNN自动学习。为了减轻在模型更新中tracker的**drifting problem**，跟踪器利用初始帧的物体标记信息以及在新获得的观测的图像的 **ground truth**外观信息。使用启发式模式判断是否更新对象外观模型。
- Hong等人[219]提出了一种基于预先训练的cnn的视觉跟踪算法，其中网络最初是为大规模图像分类而训练的，并且学习到的表示转化为描述的物体。在cnn隐藏层的顶部，他们添加了一个在线支持向量机的附加层，用于区别学习目标及其背景。通过反向投影相关的信息到输入图像空间，模型通过SVM的学习来计算特定目标的显著图 **saliency map**。他们利用目标特定的显着性图来获取生成的目标外观模型，并在了解目标空间配置的情况下进行跟踪。

---  
## **Pose Estimation姿态估计**
自从深度结构学习突破了瓶颈，许多利用CNN学习人体姿态估计的多层表达和抽象的研究越来越受重视【220，221】。
- **Deep Pose**[222]是最早将CNN应用于人体姿态估计问题的模型。在这模型中，姿态估计被视为基于CNN的人体关节坐标回归问题。模型提出一个7层瀑布型CNN以整体的方式对人体姿态进行推理。与以前那些通常显式设计图形模型局部检测器的工作不同，DeepPose通过将整个图像作为输入来捕获每个身体关节的完整上下文。

同时，也有一些工作利用CNN学习局部身体部位的表示。
- Ajrun et al. [223]提出了一种基于cnn端到端学习的人体姿态估计方法，其中cnn部分检测器和马尔可夫随机场(MRF)类空间模型被联合训练。使用卷积先验计算图中的**成对potentials(pair-wise potentials)**
- Tompson等人【224使用 **multi-resolution CNN**来计算人体各个部分的热图。与【223】不同，tomposen等人学习身体部分的先验模型并隐式的了解空间模型的结构。具体来说，他们以一对一的方式将身体的每一部分连接到自己和身体的其他部分，然后用一个完全连通的图形来建模空间先验。作为[224]的拓展，Tompson等人[92]提出了一种CNN结构，其中包含了一个经过粗略姿态估计CNN后的位置细化模型。该精化模型是一个孪生网络 （**Siamese network**） [64]，与现成的模型[224]联合训练。
- 类似[224]，Chen等人[225，226]还将图形模型与CNN相结合。他们利用CNN来学习**presense of parts及其他们的空间关系的条件概率**，这些条件概率用于图形模型的**一元和pairwise terms**。学习到的条件概率可以看作是身体姿态的低维表示。
- 还有一种名为**dual-source CNN**[227]的姿态估计方法，它整合了图形模型和整体风格。它以全身形象和局部整体观为输入。 将本地信息和上下文信息结合在一起。

除了CNN的静态图像姿态估计外，最近的研究人员还将CNN应用于视频中的人体姿态估计。
- 基于工作[224]，Jain等人[228]还将RGB特征和运动特性合并到一个 **multi-resolution CNN** 结构中，以进一步提高准确性。cnn以滑动窗口的方式进行姿态估计。CNN的输入是由rgb图像及其相应的运动特征组成的三维张量，输出是包含关节响应图 **response-map** 的三维张量。在每个响应图中，每个位置的值表示存在于该像素位置的对应节点的能量。**multi-response processing** 是通过对输入数据进行下采样并输入其到网络来实现的。

---
## **Test Detection and Recognition 文本检测和识别**
长期以来，图像中文本的识别问题得到了广泛的研究。传统上，**光学字符识别 optical character recognition**(OCR) 是人们关注的焦点。OCR技术主要是在相对受限的视觉环境(例如，干净的背景，良好的文本对齐)中对图像进行文本识别。近年来，随着计算机视觉研究中高级视觉理解的发展趋势，场景图像的文本识别已经成为人们关注的焦点[233，234]。场景图像是在无约束环境下捕获的，在无约束环境中存在大量的外观变化 **appearance variations** ，给现有的OCR技术带来了很大的困难。这种担忧可以通过使用更强大和更丰富的特征表示来缓解，比如CNN模型。在利用CNN提高场景文本识别性能的同时，提出了一些工作。这工作大致可分为三类：
1. 未经识别的文本检测和定位
2. 裁剪文本图像的文本识别
3. 集文本检测和识别为一体的端到端文本识别：

### **Text Detection 文本检测**
将CNN应用于场景文本检测的先驱作品之一是[235]。[235]使用的cnn模型学习裁剪文本图片和非文本场景图片来区分两者。给定输入的多尺度图像金字塔（**the multiscale image pyramid of the input**），然后在CNN filters生成的响应图上检测文本。
- 减少文本检测的搜索空间，Xu等人 [236]提出通过最大稳定极值区域（**Maximally Stable Extremal Regions MSER**）获得一组候选字符，并通过CNN分类过滤候选字符。
- 另一项结合MSER和CNN进行文本检测的工作是[237]。在[237]中，cnn被用来区分类似文本的MSER组件和非文本组件，通过以滑动窗口方式应用CNN，然后再进行**非最大抑制（NMS）**，可以将凌乱的文本组件分开。
- 除了文本的定位之外，还有一项有趣的工作[238]利用CNN来确定输入图像是否包含文本，而不需要知道文本的确切位置。在[238]中，使用MSER获得文本候选，然后将其传递到CNN以生成视觉特征。最后，通过在**词袋（BoW）框架**中聚合CNN特征来构建图像的全局特征。

### **Text Recognition文本识别**
- Goodfellow等人 [239]提出了一个CNN模型，该模型的最后一层具有多个softmax分类器，该模型的制定方式是，每个分类器负责在多像素输入图像的每个序列位置进行字符预测。
- 作为一种不使用**词典和字典 lexicon and dictionary**来识别文本的尝试，Jaderberg等人[240]提出了一种新的**条件随机场 CRF**类CNN模型（**a novel Conditional Random Fields (CRF)-like CNN model**），用于场景文本识别中的字符序列预测和 **bigram generation**。

### **[条件随机场](https://www.zhihu.com/question/35866596/answer/236886066)**

### **[自然语言处理中的N-Gram模型详解](https://blog.csdn.net/baimafujinji/article/details/51281816)**

新的文本识别方法以**循环神经网络 RNN**的变体来补充传统的CNN模型，以更好地建模文本中字符间的序列依赖关系。
- 在[241]中，cnn从通过滑动窗口从获得的字符级图像块中提取丰富的视觉特征，序列标记 （**sequence labelling**）通过**LSTM**【242】提取。
- [243]中提出的方法与[241]非常相似，但在[243]中，使用了词典来提高文本识别性能。

### **End-to-end Text Spotting端到端文本检测**
- 对于端到端的文本检测，Wang等人。[15]应用最初为字符分类训练的CNN模型来执行文本检测。
- 与[15]类似，[244]中提出的cnn模型允许在端到端文本检测系统的四个不同子任务之间进行特征共享：
  1. 文本检测
  2. 字符区分
  3. 字符不敏感分类
  4. bigram分类
- Jaderberg等人[245]以非常全面的方式利用CNN来执行端到端的文本查找。 在[245]中，其提出的系统的主要子任务，即文本边界框过滤，文本边界框回归和文本识别均由单独的CNN模型处理
  
---

## **Visual Saliency Detection视觉显著性检测**
图像中对显著区域的定位技术称为视觉显着性预测。它是一个具有挑战性的研究课题，有大量的计算机视觉和图像处理被这个课题所提出改进。近年来，一些工作提出了利用CNNs强大的视觉建模能力进行视觉显着性预测。**多上下文信息 multi-contextual information**是视觉显着性预测中的**关键先决条件**，目前已在大多数考虑的工作中与CNN一起使用[246-250]。 
- Wang等人[246]介绍了一种新的显着性检测算法，该算法顺序 sequentially 地利用本地上下文和全局上下文。本地上下文由CNN模型处理，该模型会在输入了本地图像patches的情况下为每个像素分配一个本地显着性值。而全局上下文（**object-level information**）由深度全连接前向反馈网络处理。
- 在[247]中，cnn参数在全局上下文模型和本地上下文模型之间共享，用于预测对象proposal中发现的超像素的显着性。
- [248]中采用的CNN模型接受了大规模图像分类数据集的预培训，然后在不同的上下文级别之间共享以进行特征提取。不同级别的上下文的CNN输出连接起来作为输入闯入一个可训练的全连接前向反馈网络中，用于显著性预测。
- 类似于[247，248]，[249]中用于显着性预测的cnn模型在三个cnn流之间共享，每个流接受不同上下文尺度的输入。
- He等人[250]导出、、提出一个空间核和一个范围核，生成两个有意义的序列作为一维CNN输入，分别描述颜色的唯一性和颜色分布。所提出的序列优于原始图像像素的输入，因为它们可以减少CNN的训练复杂性，同时能够对超像素（**superpixels**）中的上下文信息进行编码。

也有基于CNN的显着性预测方法[251–253]不考虑多上下文信息。这类方法需要依赖于CNN强有力的抽象表达能力。
- 在【251】中，一系列CNN从大量随机实例化的CNN模型中，用于生成显着性检测的良好特征。但是，在[251]中实例化的CNN模型不够深入，因为模型最大层数上限为三层。
- [252](**Deep Gaze**)通过使用5个卷积层的预训练和更深层次的cnn模型，学习一个单独的显着性模型，将来自每个cnn层的结果联合起来用于预测显著性值。
- [253]是使用cnn以端到端的方式进行视觉显着性预测的唯一工作，这意味着cnn模型接受原始像素作为输入，并生成显着性地图作为输出。潘等人[253]认为，提出的端到端方法之所以成功，是因为它的cnn结构不那么深，尝试防止过拟合。

---
## **Action Recognition行为识别**
摘要动作识别是计算机视觉研究中具有挑战性的问题之一，是对人体行为的分析，并根据其视觉外观和运动动力学对其活动进行分类[254-256]。一般来说，这个问题可以分为两大类：静态图像中的动作分析和视频中的动作分析。对于这两组，已经提出了有效的基于CNN的方法。 在这一小节中，我们简要介绍了这两个组的最新进展。

### **Action Recognition in Still Images**
[257]的工作表明，经过训练的CNN的最后几层的输出可以作为各种任务的通用视觉特征描述。[9，258]将相同的直觉用于动作识别，他们使用预先训练过的cnn倒数第二层的输出来表示动作的完整图像以及里面的人类的 **bounding box**，并在行动中达到高水平的性能。 Gkioxari等人[259]在这一框架中增加一个部分检测。他们的部分检测器是基于cnn的扩展，是对原始**Poslet**[260]方法的扩展。

在[261]中，基于CNN的上下文信息表示被用于行动识别。他们在图像中的大量对象建议proposal区域中搜索最具代表性的次要区域，并以自下而上的方式将上下文特征添加到主要区域的描述（人类主体的ground truth边界框）中。他们利用CNN来表示和微调  主要区域和上下文区域的表示,在此之后，他们向前迈出了一步，并表明在图像中定位和识别人类行为是可能的，而无需使用人类边界框[262]。在[263]中，他们提出了一种方法，该方法以最少的注释工作将人与人之间的交互作用的动作掩码分割出来。

### **Action Recognition in Video Sequences在视频序列中的动作识别**
将cnn应用于视频是很有挑战性的，因为传统的cnn是用来表示二维纯空间信号的，但是在视频中添加了一个新的时轴，与来自图像的空间变化本质上是不同的[256，264]。与图像相比，视频信号的大小也更高，这使得卷积网络的应用变得更加困难。
- Ji等人[265]建议以类似于其他空间轴的方式考虑时间轴，并引入一个用于视频输入的3D卷积层网络。
- 最近Tran等人[266]研究这种方法的绩效、效率和有效性，并显示其与其他办法相比的优势。
- 另一种将CNN应用于视频的方法是将卷积保持在2D中，并按照[267]的建议**融合连续帧的特征映射**。他们评估了三种不同的融合策略：晚期融合、早期融合和慢速融合，并将它们与CNN在单个帧上的应用进行了比较。
- 通过CNNs更好地识别行动的前向一步是，按照Simonyan和Zisserman[268]的建议，将**表示与空间和时间的变化分开**，并为每个CNN进行单独的CNN训练。此框架的第一个流是应用于所有帧的传统CNN，第二个流接收输入视频的密集光流并训练另一个CNN，该CNN在大小和结构上均与空间流相同。这两个流的输出是=将在一个类分数融合步骤中组合。
- 切隆等人[269]利用两个CNN流检测人体局部部分，并显示基于局部CNN的descriptors的聚合可以有效地提高动作识别的性能。

与空间变化不同的另一种建模视频动态的方法是将基于CNN的各个帧的特征帧送到**序列学习模块**，比如循环神经网络。Donahue等人[270]研究了在该框架中应用LSTM单元作为序列learner学习器的不同配置。

---
## **Scene Labeling场景标记**
场景标注的目的是关联一个语义类(道路、水、海等)到输入图像的每个像素[271-275]。CNN用于直接从局部图像块中模拟像素的类似然。他们能够学习强大的特征和分类器来区分局部视觉的微妙之处。
- Farabet等人[276]已率先将CNN应用于场景标记任务。他们给他们的多尺度ConvNet提供不同规模的图像片，并且他们表明学习的网络能够比具有手工制作特性的系统表现得更好。此外，该网络还成功地应用于RGB-D场景标记[277]。
- 为了使CNN具有比像素更大的视场，Pinheiro等人[278]提出了循环网络。更具体地说，在先前的迭代中，将相同的CNN循环应用于CNN的输出映射。通过这样做，它们可以获得稍微更好的标记结果，同时显着地减少推理时间（**inference time**）。
- Shuai等人[279-281]通过采样图像块训练参数化CNN （**parametric CNN**），大大加快了训练时间。他们发现基于patch的CNN存在局部模糊问题，并且 [279]通过整合**[global belief](http://papers.nips.cc/paper/5275-global-belief-recursive-neural-networks)**来解决这一问题。

[Deep belief network](https://www.sohu.com/a/167221030_714863)

- [280]和[281]利用循环递归神经网络对CNN图像特征之间的上下文依赖关系进行建模，极大地提高了标记性能。

与此同时，研究者们正在开发利用预先训练的深层CNN来进行对象语义分割。
- Mostajabi等人[282]应用ConvNet的本地和近端特性（**local and proximal features**），并应用Alex -net[8]获取远距离和全局特征，并将它们连接起来产生放大特性。它们在语义分割任务上取得了非常有竞争力的结果。
- Long等人[ 28]训练一个完全卷积的网络，直接将输入的图像预测到密集的标签地图。用ImageNet分类数据集上预先训练的模型初始化FCN的卷积层，并且反卷积层学习对标签图的分辨率进行上采样。
- Chen等人[283]还应用预先训练过的深CNN来发射像素标签。考虑到边界对齐的不完善，他们进一步利用完全连通的**CRF条件随机场**来提高标记性能。

---
## **Speech Processings**
### **Automatic Speech Recognition**
**自动语音识别(ASR)**是将人类语音转换成口语的技术[284]。在将CNN应用于ASR之前，这一领域长期以来一直被**隐马尔可夫模型和高斯混合模型(GMM-HMM)方法**所主导[285]，这通常需要在语音信号上提取手工艺特征，例如最流行的**Mel Frequency Cepstral Coefficients** (**MFCC**)特征。

与此同时，一些研究人员将深层神经网络(DNNs)应用于**大词汇量连续语音识别(LVCSR)**，并取得了令人鼓舞的结果[286，287]，但它们的网络是 在**不匹配条件 mismatch condition[288]下**易受性能退化的影响，如不同的记录条件等。

CNN比GMM-HMMs和**一般DNN**[289，290]上表现出了更好的性能，因为它们非常适合通过局部连通性来利用时域和频域的相关性，并且能够捕捉人类语音信号的频移。
- 在[289]中，它们通过在**Mel filter bank features**上应用CNN来实现较低的语音识别错误。一些尝试将原始波形与CNN一起使用，并学习滤波器以与网络的其余部分一起处理原始波形[291，292]。
- CNN在ASR中的早期应用大多只使用较少的卷积层。例如，Abdel-Hamid等人[290]在其网络中使用一个卷积层，以及Amodei等人。[293]使用三层卷积层作为特征预处理层

最近，非常深的CNNS在ASR[294,295]中显示出令人印象深刻的性能。此外，小滤波器已成功地应用于混合NN-HMM
ASR系统中的声学建模，并将池操作替换为用于ASR任务的密集连接层[296]。
- Yu等人[297]提出了一种基于注意模型的分层上下文扩展方法。它是时滞神经网络(**time-delay neural network**)[298]的一种变体，其中较低层侧重于提取简单的局部模式，而较高层则更广泛地开发上下文和提取复杂模式。类似的想法可以在[40]中找到。


### **Statistical Parametric Speech Synthesis统计参数语音合成**
除了语音识别之外，CNNS的影响还扩展到**统计参数语音合成(SPSS)**。语音合成的目的是直接从文本中产生语音，并且可能需要附加信息。与自然语音相比，浅层结构隐马尔可夫网络(**shallow structured HMM networks**)产生的语音往往是低沉的。许多研究已经采用深度学习来克服这种缺陷[299-301]。 这些方法的优点之一是它们具有强大的能力，可以通过使用生成建模框架（**generative modeling framework**）来表示内在相关性。.受神经自回归生成模型(如图像[302]和文本[303])最近进展的启发， **WaveNET**[39]利用CNN的生成模型来表示给定语言特征的声学特征的条件分布，这可以看作是SPSS的一个里程碑。为了处理长期的时间依赖关系，他们开发了一种基于扩展的因果卷积（**dilated causal convolutions**）的新体系结构，以捕获非常大的接受域。通过调节文本的语言特征，它可以直接从文本中合成语音。

---
## **Natural Language Processing自然语言处理**

### **Statistical Language Modeling**
统计语言模型：是描述自然语言内在的规律的数学模型。广泛应用于各种自然语言处理问题，如语音识别、机器翻译、分词、词性标注，等等。简单地说，**语言模型就是用来计算一个句子的概率的模型**。

对于统计语言建模，输入通常由不完整的单词序列组成，而不是完整的句子[304，305]。
- Kim等人 [304]在每个时间步使用字符级CNN的输出作为LSTM的输入。
- Gencnn[306]是用于序列预测的卷积架构，它使用单独的门控网络来替换最大池操作。
- 最近，Kalchbrenner等人[38]提出了一种基于CNN的序列处理体系结构，称为**ByteNet**，它是一个由两个扩展CNN组成的堆栈。像WaveNet [39]一样，ByteNet还受益于扩展的卷积来增加接收字段的大小，因此可以对具有长期依赖性的顺序数据进行建模。它还具有计算时间与序列长度成线性关系的优点。

与递归神经网络相比，cnn不仅可以得到远距离信息，而且还可以得到输入单词的分层表示。GU等人[307]和Yann等人[308]有一个相似的想法，那就是他们都使用CNN，而不通过池来模拟输入的单词。
- GU等人[307]将**语言CNN**与循环公路网（**recurrent highway network**）相结合，与基于LSTM的方法相比，取得了巨大的改进。
- 受LSTM网络中的选通机制的启发，[308]中的门控CNN采用门控机制来控制网络中的信息流动路径，并实现了Wikitext-103的state-of-the-art(最高水平。

然而，[308]和[307]中的框架仍然处于recurrent框架之下，其网络的输入窗口大小有限。如何捕获特定的长期依赖项以及历史词汇的分层表达仍然是一个有待解决的问题。

### **Text Classification文本分类**
文本分类是自然语言处理(NLP)的一项重要任务。自然语言句子具有复杂的结构，既有顺序结构，也有层次结构，这对于理解它们来说是必不可少的。CNNS由于捕捉时间或分层结构的局部关系的强大能力，在句子建模中达到了最高的性能。选择一个合适的CNN结构对于文本分类是非常重要的。Colobert等人[309]和Yu等人[310]使用一个卷积层来模拟句子，而Kalchbrenner等人[311]堆积多层卷积到模型句子。
- 在[312]中，它们使用多通道卷积和可变核进行句子分类。结果表明，多个卷积层有助于提取较高层次的抽象特征，多个线性滤波器可以有效地考虑不同的 **n-gram** 特征。
- 最近，Yang等人[313]通过分层卷积结构和进一步探索多通道和可变大小的特征检测器（**multichannel and variable size feature detectors.**），拓展了[312]中的模型。池化操作可以帮助网络处理可变的句子长度。
  
在[312，314]中，他们使用最大池化来保存最重要的信息来表示句子。但是，max池化无法区分其中一行中的相关特征是否仅出现一次或多次，并且它忽略了特征出现的顺序。
- 在[311]中，他们提出了k-max池化，它以输入序列的原始顺序返回前k个激活值。动态k-max池是k-max池算子的推广，其中k值取决于输入特征映射大小。

与在计算机视觉上非常成功的深CNN相比，上述CNN结构相当浅。最近，Conneau等人[315]实现了高达29个卷积层的深卷积结构。他们发现，当网络非常复杂时，快捷连接(**shortcut connection**)在高大49层的网络中会提供更好的结果。然而，在这种情况下，他们也并没有达到最先进的水平.

<br>

# **Conclusion and Outlook**

深度CNN在图像、视频、语音和文本处理等方面取得了突破性进展。在本文中，我们对CNNs的最新进展进行了广泛的综述。我们讨论了cnn在不同方面的改进。 即层设计、激活函数、损失函数、正则化、优化和快速计算。除了调查cnn各个方面的进展外，我们还介绍了CNN的应用。 介绍了cnn在图像分类、目标检测、目标跟踪、姿态估计、文本检测、视觉显着性检测、动作识别、场景标注， 语言和自然语言处理等多项任务中的应用。

虽然CNNs在实验评价方面取得了很大的成功，但仍存在许多值得进一步研究的问题。
1. 由于最近的CNN越来越深入，它们需要大规模的数据集和大规模的计算能力来进行训练。手工收集标记数据集需要大量的人力。 因此，需要探索无监督的CNN学习。
2. 同时，为了加快训练过程，虽然已经有一些使用CPU和GPU集群显示出良好效果的异步SGD算法，但仍然值得开发有效和可扩展的并行训练算法
3. 在测试时，这些深层次的模型需要很高的内存和时间消耗，这使得它们不适合部署在资源有限的移动平台上。因此如何降低复杂性并获得快速执行的模型而不损失精度也十分重要

此外，将cnn应用于新任务的一个主要障碍是需要相当多的技能和经验来选择合适的超参数，如学习速率、卷积的内核大小等。 卷积核的层数等。这些超参数具有内部依赖性，这使得它们对于调优特别昂贵。最近的工作表明，目学习深入的CNN架构前的优化技术存在很大的改进空间 [12，42，316]

最后，CNNs的坚实理论仍然缺乏。目前的CNN模型非常适合各种应用。然而，我们甚至不知道它在本质上是为什么和如何工作的。在研究CNNs的基本原则方面需要作出更多努力。同时，如何利用自然视觉机制（**natural visual perception mechanism**）进一步改进CNN的设计，也是值得探讨的问题。我们希望本文不仅能对CNNs有更好的了解，而且能为CNNs的未来研究活动和应用开发提供参考。

<br>



[从HMM到CRF到LSTM+CRF](https://blog.csdn.net/u011724402/article/details/82078328)