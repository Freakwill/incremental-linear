# 增量学习的Bayes线性回归以及在印染工业中的应用

Song, Zhang

**摘要** 工业生产活动总是不断积累数据。按照普通机器学习算法，要一次性处理所有数据，效率不高。增量学习非常适合这种情况。印染工业有一种线性模型：染料光谱与浓度存在线性关系。机器学习的任务是计算线性模型的系数。但是由于数据的积累，设备的改进，数据量会变得非常庞大，严重超出内存，而且相对于理论上的线性关系会发生一些变化。普通线性回归根本不能满足需求。为此，本文提出一种 Bayes 线性回归的增量学习算法，融合了增量学习和线性模型，可应用于印染工业。数值实验证明这个算法是成功的。

**关键字** 增量学习 Bayes线性回归 印染工业 证据逼近过程 边缘极大似然



# Incremental leaning for Bayesian linear regression and its application in dying industry

Song Congwei1 ，Zhang Xiaoming2

1Yanqi Lake Beijing Institute of Mathematical Sciences and Applications  101408
2Yanqi Lake Beijing Institute of Mathematical Sciences and Applications  101408

**Abstract** Industrial activity is always accumulating data. According to the ordinary machine learning algorithm, it is not efficient to process all the data at once. Incremental learning is perfect for this situation....



**Keywords** I incremental learning；Bayesian linear regression；dyeing industry；evidence approximation procedure；marginal maxi-mum likelihood



## 引言

工业生产活动总是在不断积累数据。按照普通机器学习算法，要一次性处理所有数据，效率不高。一旦有新的数据产生通常就要集合到旧数据中重新计算。以印染工业为例，为了能够得到预期的光谱属性（布料被染成什么颜色）确定适合的染料浓度配比，人们积累了大量数据，而且每天都在增加。和科学常识基本相符，光谱属性和染料浓度配比是呈线性关系的。但是由于数据的积累，设备的改进，数据会有偏差，也会发生微妙的变化。实际浓度配比和理论值有出入。工人不得不在正式染色前，先对样品做试验。普通线性回归在某些时候能给出精确预测，减轻手工操作，但是一次性处理所有数据（已经学习过的和新获得的）是低效的，否则要么不去利用新数据，要么丢掉部分旧数据重新学习，不能在旧模型和新数据之间保持平衡。为了提升机器学习的性能，适应工业生产活动的需要，本文提出一种建立在 Bayes 线性回归基础上的增量学习算法，结合了增量学习和线性模型，应用于印染工业。

增量学习可以看成是一种能够减少内存的批量学习：一方面数据可以按顺序分批学习，另一各方面学习过的数据可不再保存。它也是一种特殊的终身学习：前一次学习结果会影响下一次学习，只是学习任务来自于单个问题。本文提出的增量Bayes 线性回归是一种简单实用的机器学习算法，适合解决线性问题。

记号说明$p(x|y)$表示条件概率。$\sum_i$是求和符号。$A^T$表示矩阵转置。$a\cdot b$表示（列）向量内积运算，即$a^Tb$。$a\circ b$表示按位相乘（Hardmard乘法）。数量和矩阵相加是对应的数量矩阵和矩阵相加,如*σ*+*A*=*σI*+*A*。用绝对值符号|*a*|表示向量Euclid范数。

## 理论基础

### Bayes 线性模型

用x表示输入变量，y表示输出变量，在本文分别表示染料光谱和浓度。一般的线性模型可以写成
$$
y=\sum_iw_i\phi_i(x)+\epsilon\\
=w\cdot\phi(x) + \epsilon, \epsilon\sim N(0,\sigma^2)
$$
其中$\phi(x)=(\phi_1,\cdots,\phi_p)$是一系列基函数, 本文根据事实，简单选用$\phi_j(x)=x_j$和一个常数项，即$y$与$x$存在线性关系.  观察到的样本集$(x_i,y_i), i=1,\cdots,N$，应该满足$y=\Phi w +c$，其中$\Phi=\{\phi_j(x_i)\}$是$N\times p$的“设计矩阵”。在普通线性回归中，人们求解下述法方程(normal equation)
$$
Gw=b, G=\Phi^T\Phi,b=\Phi^Ty
$$
和线性代数一样，称$G$为基向量$\Phi$产生的 Gram 矩阵。Gram 矩阵通常只和输入变量有关，也不包含任何参数，会被事先计算好。这样既方便了计算，在内存紧缺的情况下甚至也不需要存储数据集。不过，这要设计新算法来支持。

即使是最简单的线性回归，处理海量的数据依然无法满足需求。如果旧数据被整合到新数据中学习，那么效率会非常低；如果只学习新数据，又不能充分利用旧数据的学习成果。当然更不能只利用旧数据的学习成果。引言提到用增量学习满足工业上的需求。Bayes 方法就是一种天然的增量学习方法，它的一般形式为
$$
p(\theta |D, D') =\frac{p(D'|\theta)}{\int p(\theta|D)p(D'|\theta)\mathrm{d}\theta}p(\theta|D)
$$
其中$D$是已经被学习过的数据集，$D'$是新数据集。该公式避免直接计算$p(\theta |D, D')$，而是充分利用$p(\theta |D)$，这是参数后验分布，但在(2)中成了一个改进版的先验分布。一般情况下，人们也很少直接用(2)。在某些场合下，如Bayes线性回归，可以导出一些高效的迭代公式。

在Bayes模型中，合理假设参数的先验分布
$$
w_i\sim N(0,\alpha_i^{-1})
$$
超参数$\alpha_i$ ($\alpha_i^{-1}$)还服从比如 (逆) $\Gamma$分布，但本文简化为无信息先验，包括$\sigma^2$。令$\alpha=(\alpha_1,\cdots,\alpha_p)^T$, 按位求倒数$\alpha^{-1}=(\alpha_1^{-1},\cdots,\alpha_p^{-1})^T$。方差$\alpha^{-1}$的写法，对本文来说，不是很友好，但是笔者遵循习惯记法。

*注*和多数工业数据一样，本问题中，$y$是多维的，但是本文依然独立处理每个分量。这种独立性体现在每个样本每个（$y$的）分量之间误差项$\epsilon$始终是独立的。因此本质上还是一维的。

### 证据逼近过程(evidence approximation procedure)

在超参数的无信息先验假设下，Bayes 线性模型的参数$w$服从$N(\mu,\Sigma)$，其中
$$
\mu = \sigma^{-2}\Sigma \Phi^T y= \sigma^{-2}\Sigma b\\
\Sigma =(A+\sigma^{-2}\Phi^T\Phi)^{-1}= (A+\sigma^{-2}G)^{-1}, A=\mathrm{diag}(\alpha)
$$
$\mu$作为$w$的估计值。然后就可以对新输入值$x'$预测$y(x’)=\phi^T(x')\mu_i$。置信度由方差$\sigma^2+\phi^T(x')\Sigma\phi(x')$决定，其中$\phi(x)=(\phi_1(x'),\cdots,\phi_p(x'))^T$。

超参数$\alpha,\sigma^2$通过边缘极大似然估计计算。人们以及开发了高效迭代算法——证据逼近过程。

算法1

准备$b=\Phi^Ty,G=\Phi^T\Phi,q=\|{y}\|^2$

初始化$\alpha,\sigma^2$

循环计算：

1. 根据(5)计算$\mu,\Sigma$
2. $\gamma_i\leftarrow 1-\alpha_i\Sigma_{ii}$
3. $\alpha_i\leftarrow\frac{\gamma_i}{\mu_i^2}$
4. $\beta\leftarrow\frac{N-\sum_i\gamma_i}{\|\Phi \mu-y\|^2}=\frac{N-\sum_i\gamma_i}{\mu^TG\mu-2\mu^T b+q}$

输出$\alpha,\sigma^2=\frac{1}{\beta}$

这是EM算法的迭代风格，1是E步，2-4是M步。

由于可能会出现$\alpha_i\to \infty$，导致溢出，本文把算法1改造为更为稳定的算法2。算法2的特点是直接计算$\alpha^{-1},\sigma^{2}$，分别是系数和误差的方差，而不是它们的倒数。

引入
$$
\mu = \triangle A^{-1}b=\triangle (\alpha^{-1} \circ b)\\
\triangle = (\sigma^2+A^{-1}G)^{-1}\\
\square = \sigma^2\triangle\\
\Sigma = \square A^{-1}
$$

不难证明$\alpha_i\Sigma_{ii}=\square_{ii}$，且由线性代数的知识可得
$$
0\leq \Sigma_{ii}''\leq 1, \Sigma_{ii}''\to 1, \alpha_i\to\infty
$$
因此当$\gamma_i$非常小时，继续迭代$\alpha^{-1}_i$已经没有意义了, 可直接令$\alpha^{-1}_i=0$.

算法2

准备$b=\Phi^Ty,G=\Phi^T\Phi,q=\|{y}\|^2$

初始化$\alpha^{-1},\sigma^2$

循环计算，直到$\sigma^2$ 很小或者满足其他条件：
1. 根据(6)计算$\mu,\Sigma$，以及辅助变量$\triangle, \square$

2. $\gamma_i\leftarrow 1-\square_{ii}\in [0,1]$

3. $\alpha_i^{-1}\leftarrow \frac{\mu_i^2}{\gamma_i}$，当$\gamma_i=0$(或很小)时, $\alpha^{-1}_i=0$

4. $\sigma^{2}\leftarrow \frac{\mu^TG\mu-2\mu^T b+q}{N-\sum_i\gamma_i}$


返回 $\alpha^{-1},\sigma^2$

*注* $\sigma^2$可以代表训练误差，因此当它很小时，可以停止迭代。

### Bayes线性回归的增量学习算法

节论述设计增量学习算法的基本原理，它是证据逼近的增量学习版本。不过，在正式设计算法前，先要处理一下数据溢出的问题。

设新数据集为$(X', Y')$, 对应的Gram矩阵为$G'$，令$b'=\Phi'^Ty'$, 增广数据集$(\tilde{X},\tilde{Y})=((X,X'),(Y,Y'))$, 对应地有
$$
\tilde{\Phi}=\begin{pmatrix}\Phi\\\Phi'\end{pmatrix}, \tilde{y}=\begin{pmatrix}y\\y'\end{pmatrix},\\
\tilde{G}=G+G', \tilde{b}=b+b', \tilde{q}=q+q',
$$
相关参数记为$\tilde{\alpha}, \tilde{\sigma},\tilde{A}=\mathrm{diag}\{\tilde{\alpha}\}$等等。我们可以直接利用(5)和算法1计算$\tilde{\alpha}, \tilde{\sigma}$，初始值可以用之前的计算输出。然而，实际上$\tilde{G}=G+G'$这类求和运算会导致溢出，因为这本质上是一个超高维度向量的内积运算。

即使是第一次训练也要谨慎使用。均值化处理，是简单有效的手段。若引入变量$b=\frac{\Phi^Ty}{N},G=\frac{\Phi^T\Phi}{N},q=\frac{\|{y}\|^2}{N}$，即分别用$\frac{\Phi}{\sqrt{N}},\frac{y}{\sqrt{N}}$代替$\Phi,y$, 则模型变成

$$
\frac{\Phi}{\sqrt{N}}w=\frac{y}{\sqrt{N}}+N(0,\frac{\sigma^2}{N})
$$

算法2最终的返回值是$\alpha^{-1},\frac{\sigma^{2}}{N}$。因此，循环结束存储$\sigma^2\leftarrow N\sigma^2$, 这才是原问题超参数的值。随着$N$的增大，且$\sum_i\gamma_i< p$，可令$\sigma^{2}\leftarrow \frac{\mu^TG\mu-2\mu^T b+q}{N}$，即平方误差的均值，可以作为模型的误差。


等效的做法是把迭代的第3, 4步改为
$$
\alpha_i^{-1}\leftarrow \frac{N\mu_i^2}{\gamma_i}, \sigma^{2}\leftarrow \frac{\mu^TG\mu-2\mu^T b+q}{1-\sum_i\gamma_i/N}
$$
返回值一定是$\alpha^{-1},\sigma^{2}$。

随着$N$的增大，而$\sum_i\gamma_i<p$有界，可令
$$
\sigma^{2}\leftarrow \mu^TG\mu-2\mu^T b+q
$$

*注* 算法中的变量不一定存储名义上代表的参数数值。


下面，着手建立增量的证据逼近算法。令新数据比例$r=\frac{N'}{N+N'}$。$r$值通常会越来越小，它也可以人为设定，尤其当旧数据不能给予太大权重时。

一种过于简单的增量学习方案是，将$(1-r)G +rG', (1-r)b +rb', (1-r)q+rq'$代入算法，初始化$\alpha^{-1},\sigma^2$为先前迭代过程的返回值。这种方案虽然比较精确，也有一定可行性，但并没有充分利用第一次迭代的结果。

下面要给出另一种增量学习方案：一种快速但近似化的更新方法。这也是本文最主要的贡献。

合理假定已获得$\alpha^{-1}$的精确解，因此$w$的先验分布（相对于新数据）是$N(\mu,\Sigma)$, 后验分布（由Bayes公式得到）是$N(\mu',\Sigma')$，其中
$$
\mu' = \Sigma'(\sigma^{-2}b'+\Sigma^{-1}\mu)=(\sigma^2+\Sigma G')^{-1}(\Sigma b'+\sigma^2\mu)\\
\Sigma' =(\Sigma^{-1}+\sigma^{-2}G')^{-1}=\sigma^2(\sigma^2+\Sigma G')^{-1}\Sigma
$$
注意，我们依然不知道$w$的精确值。

可对$\sigma^2$做一次近似的调整
$$
\sigma^{2}\leftarrow (1-r)\sigma^{2}+r\sigma'^2,
$$
其中$\sigma'^2=\mu'^TG’\mu'-2\mu’^T b’+q'$。这个方案的优点是，不需要迭代，只要做一步更新，无需均值化处理。(12)中就没有做均值化处理。

正常情况下，$w$方差$\Sigma$会越来越小，$\sigma^2+\Sigma G'$的可逆性最终依赖于$\sigma^2$。如果$\sigma^2$也很小，就不必做任何更新了，而此时模型误差也很小了。

最后补充一点。根据$\alpha^{-1}$可以判断哪些属性（染料）并不重要。当$\alpha_i^{-1}$很小时，可以认为$w_i=0$，剔除第$i$个属性。新数据一般也不会增加其权重。这样做可以减少计算量，也可以消除过拟合现象，另外，也有重要的现实意义，至少可以少测量一种染料光谱值。

## 算法设计
根据上文提到的增量学习方案，我们设计增量Bayes线性回归算法，并用计算机语言实现。

### 算法设计与注意事项

根据上文的讨论，我们设计增量Bayes线性回归算法。

算法3

根据算法2学旧数据

输入新数据$(X',Y')$的$N'$个观测值。

初始化$\sigma^2,\alpha$，其初始值是上一阶段训练的输出值

初始化$ \mu,\Sigma$（上一阶段训练完毕时存储，无需再次计算）

准备$G\leftarrow \Phi'^T\Phi', b\leftarrow \Phi'^Ty',q\leftarrow \|y'\|^2$

单步更新：

1. 根据(12)更新$\mu,\Sigma$
2. 根据(13)更新$\sigma^2$

返回$\alpha^{-1},\sigma^2$

注意，本算法中，没有用到旧数据，包括上一步的 Gram 矩阵。

### 算法实现

算法通过Python3实现，主要依赖scikit-learn。实验环境为macOS10.15，Python3.8，scikit-learn0.24.1。源代码已经上传到GitHub。

## 数值实验

数据来自一家绍兴印染企业。我们把数据分成三部分，分别代表旧训练数据、新训练数据、测试数据，各占总数据的48%,32%和20%。

增量模型总是先学习旧数据再学习新数据，每次都会对测试数据进行测试，得到测试分数1和测试分数2。在只学习了旧数据和学习了新数据后，分别有测试新数据和旧数据的机会。实验都利用了这两个机会，先后获得新数据测试分数和旧数据测试分数。只学习旧（新）数据的模型没有旧（新）数据的测试分数，但可以测试新（旧）训练数据和测试数据。所有一次性学习算法也有两个阶段，先单独学习旧数据并对预测测试数据，然后学习所有训练数据并再一次做预测；且其旧数据训练分数是指第一阶段的训练分数，而新数据训练分数是第二阶段产生的；和增量模型一样，产生两个测试分数。

| 模型           | 旧数据训练分数 | 旧数据测试分数 |   新数据训练分数  | 新数据测试分数 |    测试分数1 | 测试分数2 | 耗时1 | 耗时2 |
| --------------------------- | -------------- | -------------- | ---- | -------------- | ---- | -------- | -------- | --------------------------- |
| 增量Bayes线性回归     |                |                |      |                |      |  |  |  |
| Bayes线性回归(只学习旧数据) |                | 无 | 无 |              |      |  |  |  |
| Bayes线性回归(只学习新数据) | 无             |              |      | 无 |      |  |  |  |
| Bayes线性回归(一次性学习)   |                | 无             |      | 无 |      |  |  |  |
| 普通线性回归(一次性学习)    |                | 无 |      | 无 |      |  |  |  |
| Bayes脊回归 | |  | |  | |  |  |  |

每个模型的数值实验都重复了20次，每次都按给定比例对数据进行分配。表中每个空格所填数值是重复试验得到的20个数值的中位数。

实验结果非常符合预期。比如学习新数据后，增量模型的测试分数提高了，而旧数据测试分数有所减小，毕竟旧模型理论上相对于旧数据是最优的线性模型。单独学习的性能总是不如增量模型，而一次性学习虽然分数高，但是耗时长，而且也意味着较大的内存开销，另外除了有较高的训练分数，会表现出轻微的过拟合。和增量模型相比，较为复杂的Bayes脊回归也表现出轻微的过拟合。

## 结语

本文算法满足增量学习的几个特点，包括批量学习，低内存需求。另外除了首次训练，其余训练都只有单步调整，速度极快，非常适合在对时间要求苛刻的工业领域应用。

实验证明本问算法的有效性，也证实了增量学习会是未来工业处理大数据的主要机器学习范式。

本文一个明显的简化在于超参数的无信息先验假设。未来的研究特别着重于超参数的分布。相应的增量学习也会被开发出来。而本文算法由于其思路和设计的简单性，将在工业上长久发挥作用。另外，误差项不一定是服从独立的正态分布。由于数据充分多，可以考虑参数更多的后尾分布，如学生$T$分布。不同的分布往往意味着不同的损失函数。

此外，随着数据的基类，新数据比例$r$会趋于0，不利于学习。除了可以手动提高$r$值，我们会考虑其他更好的平衡旧模型和新数据的策略。本算法还没有采用核函数。只要应用上有需要，可以立刻启用核函数。





**参考文献**

杨晓伟 郝志峰 支持向量机的算法设计与分析【M】北京：科学出版社2013。Yang X Hao Z Algorithm Design and Analysis of SVM143-147

Walter G., Augustin T. (2010) Bayesian Linear Regression — Different Conjugate Models and Their (In)Sensitivity to Prior-Data Conflict[M]// Kneib T., Tutz G. Statistical Modelling and Regression Structures. Physica-Verlag HD. 2009: 59-78

Fletcher T. Relevance Vector Machines Explained 2010. http://home.mit.bme.hu/~horvath/IDA/RVM.pdf

Hastie T, Trevor R, Tibshirani J. THE ELEMENTS OF STATISTICAL LEARNING: DATA MINING, INFERENCE, AND PREDICTION, SECOND EDITION[M]. Springer, 2001.44-49, 272-279



M. Sugiyama, Introduction to Statistical Machine Learning  Elsevier (Singapore),2016. 94-95, 162-166.



M E Tipping Sparse Bayesian Learning and the Relevance Vector Machine[J]. Journal of Machine Learning Research, 2001, 1: 211-244.



Tzikas D., Likas A., Galatsanos N. (2008) Incremental Relevance Vector Machine with Kernel Learning[C]. In: Darzentas J., Vouros G.A., Vosinakis S., Arnellos A. (eds) Artificial Intelligence: Theories, Models and Applications. SETN 2008. Lecture Notes in Computer Science, vol 5138. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-87881-0_27



Laskov P Gehl C, Kruger S, Muller K R. Incremental support Vector Machine learning: analysis, implementation and application[J]. Journal of Machine Learning Research, 2006, 7:1909-1936.

---

## 无效片段

而$\sigma^2$是下述最大化问题的解
$$
\max_{\sigma^2} \ln \prod_ip(y_i|x_i,\sigma^2)\sim -\frac{N}{2}\ln\sigma^2 +\frac{(b^T\Sigma+\mu^T)(\sigma^2\Sigma+\Sigma G\Sigma)^{-1}(\Sigma b+\mu)-q}{\sigma^2}
$$
解得

增广后的数据也对应的参数也应满足：
$$
\tilde{\mu}= (\tilde{\sigma}^2+\tilde{A} ((1-r)G+rG')^{-1}\tilde{A}^{-1} ((1-r)b +rb')\\
\tilde{\Sigma} = \tilde{\sigma}^2(\tilde{\sigma}^2+\tilde{A} ^{-1}((1-r)G+rG'))^{-1}\tilde{A}^{-1}
$$
令$\tilde{\alpha}= \alpha$（也是一种近似）, 并给出近似$\tilde{\sigma}^{2}\sim(1-r)\sigma^{2}+r\sigma'^2$, 则有下述近似
$$
\tilde{\Sigma} \sim \tilde{\sigma}^2((1-r)(\sigma^2+AG)+r(\sigma'^2+AG')))^{-1}A^{-1}\\
\sim \tilde{\sigma}^2((1-r)+r\triangle(\sigma'^2+A^{-1}G')))^{-1}\triangle A^{-1}\\
\tilde{\mu}\sim((1-r)+r\triangle(\sigma'^2+A^{-1}G')))^{-1}\triangle A^{-1}((1-r)b +rb')\\
\sim((1-r)+r\triangle(\sigma'^2+AG')))^{-1}((1-r)\mu +r\triangle A^{-1}b')
$$

迭代次数不能过多，一方面这里使用了近似，另一方面，大数据条件下，第一次训练收敛很快，每次增量学习都单步更新就足够了。本文推荐并采用单步调整。

