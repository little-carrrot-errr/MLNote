<!-- TOC -->

- [**RGB颜色空间**](#rgb%e9%a2%9c%e8%89%b2%e7%a9%ba%e9%97%b4)
- [**HSV(Hue, Saturation, Value)**](#hsvhue-saturation-value)
  - [色调H](#%e8%89%b2%e8%b0%83h)
  - [饱和度S](#%e9%a5%b1%e5%92%8c%e5%ba%a6s)
  - [明度V](#%e6%98%8e%e5%ba%a6v)
  - [**HSV颜色空间**](#hsv%e9%a2%9c%e8%89%b2%e7%a9%ba%e9%97%b4)
  - [**RGB和HSV转换**](#rgb%e5%92%8chsv%e8%bd%ac%e6%8d%a2)

<!-- /TOC -->
<br>

- [百度百科](https://baike.baidu.com/item/HSV/547122?fr=aladdin)
- [参考博客](https://blog.csdn.net/kakiebu/article/details/79476235)
- [RGB、YUV和HSV颜色空间模型](https://www.cnblogs.com/justkong/p/6570914.html)

## **RGB颜色空间**

1. 计算机色彩显示器和彩色电视机显示色彩的原理一样，都是采用R、G、B相加混色的原理，通过发射出三种不同强度的电子束，使屏幕内侧覆盖的红、绿、蓝磷光材料发光而产生色彩。这种色彩的表示方法称为RGB色彩空间表示。

2. 在RGB颜色空间中，任意色光F都可以用R、G、B三色不同分量的相加混合而成：**F=r[R]+r[G]+r[B]**。RGB色彩空间还可以用一个三维的立方体来描述。当三基色分量都为0(最弱)时混合为黑色光；当三基色都为k(最大，值由存储空间决定)时混合为白色光。
![](https://images2015.cnblogs.com/blog/536358/201703/536358-20170318083044510-544926603.jpg)

3. RGB色彩空间根据每个分量在计算机中占用的存储字节数分为如下几种类型：
    
    (1) RGB555

    RGB555是一种16位的RGB格式，各分量都用5位表示，剩下的一位不用。

    高字节 -> 低字节：X RRRRR GGGGG BBBBB

    (2) RGB565

    RGB565也是一种16位的RGB格式，但是R占用5位，G占用6位，B占用5位。

    (3) RGB24

    RGB24是一种24位的RGB格式，各分量占用8位，取值范围为0-255。

    (4)RGB32

    RGB24是一种32位的RGB格式，各分量占用8位，剩下的8位作Alpha通道或者不用。

4. RGB色彩空间采用物理三基色表示，因而物理意义很清楚，适合彩色显象管工作。然而这一体制并不适应人的视觉特点。因而，产生了其它不同的色彩空间表示法。


## **HSV(Hue, Saturation, Value)**

是根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。

### 色调H
用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°,品红为300°；

![](https://img-blog.csdn.net/20160526140055291?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


### 饱和度S
饱和度S表示颜色接近光谱色的程度。一种颜色，可以看成是某种光谱色与白色混合的结果。其中光谱色所占的比例愈大，颜色接近光谱色的程度就愈高，颜色的饱和度也就愈高。饱和度高，颜色则深而艳。光谱色的白光成分为0，饱和度达到最高。通常取值范围为0%～100%，值越大，颜色越饱和。

### 明度V
明度表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，此值和物体的透射比或反射比有关。通常取值范围为0%（黑）到100%（白）。

![](https://img-blog.csdn.net/20160526141816845?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

目前在计算机视觉领域存在着较多类型的颜色空间(color space)。HSL和HSV是两种最常见的圆柱坐标表示的颜色模型，它重新影射了RGB模型，从而能够视觉上比RGB模型更具有视觉直观性。

### **HSV颜色空间**
HSV(hue,saturation,value)颜色空间的模型对应于圆柱坐标系中的一个圆锥形子集，圆锥的顶面对应于V=1. 它包含RGB模型中的R=1，G=1，B=1 三个面，所代表的颜色较亮。色彩H由绕V轴的旋转角给定。红色对应于 角度0° ，绿色对应于角度120°，蓝色对应于角度240°。在HSV颜色模型中，每一种颜色和它的补色相差180° 。 饱和度S取值从0到1，所以圆锥顶面的半径为１。HSV颜色模型所代表的颜色域是CIE色度图的一个子集，这个 模型中饱和度为百分之百的颜色，其纯度一般小于百分之百。在圆锥的顶点(即原点)处，V=0,H和S无定义， 代表黑色。圆锥的顶面中心处S=0，V=1,H无定义，代表白色。从该点到原点代表亮度渐暗的灰色，即具有不同 灰度的灰色。对于这些点，S=0,H的值无定义。可以说，HSV模型中的V轴对应于RGB颜色空间中的主对角线。 在圆锥顶面的圆周上的颜色，V=1，S=1,这种颜色是纯色。HSV模型对应于画家配色的方法。画家用改变色浓和 色深的方法从某种纯色获得不同色调的颜色，在一种纯色中加入白色以改变色浓，加入黑色以改变色深，同时 加入不同比例的白色，黑色即可获得各种不同的色调。 

![](https://img-blog.csdn.net/20160526140048523?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


![](https://img-blog.csdn.net/20160526140102525?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### **RGB和HSV转换**

1. 从RGB到HSV
   
   设max等于r、g和b中的最大者，min为最小者。对应的HSV空间中的(h,s,v)值为：

   ![](https://images2015.cnblogs.com/blog/536358/201703/536358-20170320141419846-1299521890.png)

   h在0到360°之间，s在0到100%之间，v在0到max之间。

2. 从HSV到RGB

    ![](https://images2015.cnblogs.com/blog/536358/201703/536358-20170320141740721-1201160535.png)