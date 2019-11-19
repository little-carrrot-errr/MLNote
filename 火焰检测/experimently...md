- [参考](https://blog.csdn.net/qq_39273781/article/details/94546939)

# **Experimentally Defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection**

论文成果：实时了视频实时火灾监测，且速度大大提升，准确率高达93%，在superpixel上定位准确度可达到89%（超像素就是把一幅原本是像素级(pixel-level)的图，划分成区域级(district-level)的图）

构建了两种火焰检测网络
1. fireNet
   ![](https://img-blog.csdnimg.cn/20190703092225233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MjczNzgx,size_16,color_FFFFFF,t_70)
2. inception V1-OnFire
   ![](https://img-blog.csdnimg.cn/201907030923160.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MjczNzgx,size_16,color_FFFFFF,t_70)

3. Superpixel-InceptionV1-OnFire

    结构和InceptionV1-OnFire相同，只是先将图像用简单线性迭代聚类（SLIC）分为很多超像素块，再用训练好的检测模型对每个超像素块进行检测，达到FireNet和InceptionV1-OnFire都不能做到的火焰超像素区域识别

该论文主要是通过实验改善了网络模型结构，从而达到降低模型参数，给模型加速的目的，但在火焰检测准确率方面和经典网络相差无几。

在火焰区域检测方面和传统的检测方法没有区别，都是用**简单线性迭代聚类（SLIC）算法**得出火焰区域。