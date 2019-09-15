- [广义线性模型](https://blog.csdn.net/weixin_37140379/article/details/82289704)

- [关于 CNN对图像特征的 位移、尺度、形变不变性的理解](https://blog.csdn.net/voxel_grid/article/details/79275637)

- [shit-invariance](https://www.cnblogs.com/fydeblog/p/11083664.html) $\quad$  [参考视频](https://www.bilibili.com/video/av63925068)
  - 什么是平移等方差（Shift-equivariance）？ [参考](https://www.cnblogs.com/fydeblog/p/11083664.html)
    >答：$Shift {\Delta h, \Delta w}(\widetilde{\mathcal{F}}(X))=\widetilde{\mathcal{F}}\left(\text { Shift }{\Delta h, \Delta w}(X)\right) \quad \forall(\Delta h, \Delta w)$，可以看到输入在$(\Delta h, \Delta w)$变化，输出对应的输出在$(\Delta h, \Delta w)$变化。

  - 什么是平移不变性（Shift-invariance）？

    >答：$\widetilde{\mathcal{F}}(X)=\widetilde{\mathcal{F}}\left(\text { Shift }_{\Delta h, \Delta w}(X)\right) \quad \forall(\Delta h, \Delta w)$， 输入在$(\Delta h, \Delta w)$变化，不改变最后的结果。

    大多数现代的卷积网络是不具有平移不变性的（如上所示，右边是作者提出的方法BlurPool），而不具有平移不变性的原因是因为maxpooling，strided-convolution以及average-pooling这些下采样方法忽略了抽样定理

- 平铺卷积网络 Tiled Convolution $\quad$[参考1](https://blog.csdn.net/xiao_jiang2012/article/details/9349955) && [2](https://blog.csdn.net/zhq9695/article/details/84959472)

-  [图像卷积与反卷积 Transposed Convolution](https://blog.csdn.net/qq_38906523/article/details/80520950) $\quad$ [CNN中的卷积和反卷积](https://blog.csdn.net/sinat_29957455/article/details/85558870)
  
-  [空洞卷积 Dilated Convolution](https://www.jianshu.com/p/f743bd9041b3)

- [Network In Network——卷积神经网络的革新](https://www.cnblogs.com/yinheyi/p/6978223.html)