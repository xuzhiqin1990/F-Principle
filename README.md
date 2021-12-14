# 傅立叶变换

## 一维空间的傅立叶变换
* 连续情形
$$
\hat{f}(\xi)=\int_{-\infty}^{+\infty} f(x) e^{-i \xi x} d x
$$
* 离散情形
$$\hat{x}(k)=\sum_{n=0}^{N-1} x(n) e^{-i \frac{2 \pi}{N} k n},(k=0,1,2 \ldots N-1)$$
这里的 $\xi$ 和 k 就是频率
## 高纬空间的傅立叶变换
* 连续情形
$$\hat{f}(\vec{\xi})=\int_{-\infty}^{+\infty} f(\vec{x}) e^{-i \vec{\xi} \cdot \vec{x}} d \vec{x}$$
* 离散情形
$$
\hat{y}(\vec{k})=\frac{1}{n} \sum_{i=0}^{n-1} y_{i} e^{-i 2 \pi \vec{k} \cdot \overrightarrow{x_{i}}}
$$

高纬空间如何查看一个函数的低频与高频，
* 一种方法是投影，选择一个方向，在频率空间里投影，查看频率空间在某一特定方向的频率
* 计算某个球体中的能量和当成低频
  * 低频: $|\xi| \leq k_{0} \Leftrightarrow \xi_{1}^{2}+\xi_{2}^{2}+\ldots+\xi_{d}^{2} \leq k_{0}^{2}，\hat{f}(\xi)^{\text {low }}=\hat{f}(\xi) \cdot \mathcal{X}_{|\xi| \leq \xi_{0}}$
  * 高频: $|\xi|>k_{0} \Leftrightarrow \xi_{1}^{2}+\xi_{2}^{2}+\ldots+\xi_{d}^{2}>k_{0}^{2}，\hat{f}(\xi)^{h i g h}=\hat{f}(\xi) \cdot \mathcal{X}_{|\xi|>\xi_{0}}=\hat{f}(\xi)-\hat{f}(\xi)^{\text {low }}$
## 高频与低频的误差
记 $\hat{y}(k)$ 为真实值，$\hat{h}(k)$ 是$\{x_{i}, h(x_{i}))\}$ 离散傅立叶变换的结果，h 为神经网络拟合的函数。

因此，我们可以定义低频和高频的收敛误差
$$
e_{\text {low }}=\left(\frac{\sum_{k} \mathcal{X}_{|k| \leq k_{0}}|\hat{y}(k)-\hat{h}(k)|^{2}}{\sum_{k} \mathcal{X}_{|k| \leq k_{0}}|\hat{y}(k)|^{2}}\right)^{\frac{1}{2}} \quad e_{\text {high }}=\left(\frac{\sum_{k}\left(1-\mathcal{X}_{|k| \leq k_{0}}\right)|\hat{y}(k)-\hat{h}(k)|^{2}}{\sum_{k}\left(1-\mathcal{X}_{|k| \leq k_{0}}\right)|\hat{y}(k)|^{\frac{1}{2}}}\right.
$$

## 使用高斯函数来代替示性函数的傅立叶变换
低频: $\hat{f}(\xi)^{l o w}=\hat{f}(\xi) \cdot \hat{G}(\xi)$

高频: $\hat{f}(\xi)^{h i g h}=\hat{f}(\xi) \cdot(1-\hat{G}(\xi))=\hat{f}(\xi)-\hat{f}(\xi) \cdot \hat{G}(\xi)$
