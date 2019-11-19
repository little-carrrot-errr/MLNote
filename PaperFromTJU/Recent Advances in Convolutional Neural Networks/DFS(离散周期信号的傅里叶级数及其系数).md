## **[离散周期信号的傅里叶级数及其系数（DFS）](https://blog.csdn.net/Reborn_Lee/article/details/80798926)**

1. 针对对象：周期离散序列，设周期为N
2. 类似连续周期信号那样用傅里叶级数表示信号，周期序列x[n]的傅里叶级数DFS）表示：
    > <font size=4px> $x[n]=\frac{1}{N}\sum^{N-1}_{n=0}X^\sim_{[k]}e^{j\frac{2\pi kn}{N}} =\frac{1}{N}\sum^{N-1}_{n=0}X^\sim_{[k]}W^{-kn}_N$</font>
     
    ><font size=4px>$W^{kn}_N = e^{-j\frac{2\pi}{N}}$</font>

    从上面的公式中可以看到，积分限从0到N-1，而非像连续周期信号的傅里叶级数那样，从到，这是为什么呢？也就是说，为什么不像连续周期信号的傅里叶级数一样，需要无穷多个成谐波关系的复指数合成？

    这是因为：

    <font size=4px>$e^{j\frac{2\pi k(n+N)}{N}} = e^{j\frac{2\pi kn}{N}}$</font>

    即对于n来说，是以N为周期的，所以只需要一个周期就可以了。
    （连续周期信号的傅里叶变换要不要贴出来呢？）

3. 下面是**傅里叶系数**的表达式
    > <font size=4px> $x[n]=\sum^{N-1}_{n=0}X^\sim_{[n]}e^{-j\frac{2\pi kn}{N}} =\sum^{N-1}_{n=0}X^\sim_{[n]}W^{kn}_N$</font>
    
    自变量取值方位$[-\infty,\infty]$

4. **离散傅里叶变换（DFT）**
   - 正变换DFT
     > <font size=4px> $x[n]=\sum^{N-1}_{n=0}X^\sim_{[n]}e^{-j\frac{2\pi kn}{N}} =\sum^{N-1}_{n=0}X^\sim_{[n]}W^{kn}_N \quad k=0,...,N-1$</font>
   - 逆变换IDFT
     > <font size=4px> $x[n]=\frac{1}{N}\sum^{N-1}_{k=0}X^\sim_{[k]}e^{j\frac{2\pi kn}{N}} =\frac{1}{N}\sum^{N-1}_{n=0}X^\sim_{[k]}W^{-kn}_N \quad n=0,...,N-1$</font>
    
        ![](/DFT_IDFT.png)
    - 比DFS与DFT可以很明显的看到，二者之间的关系为：
      - 除了取值范围不同，其他基本一致，实际应用中，要处理的信号大多数为有限长的非周期信号，因此DFT更常用。
      - DFT只不过是特殊的DFS，就是对DFS的时域和频域只取主值部分。