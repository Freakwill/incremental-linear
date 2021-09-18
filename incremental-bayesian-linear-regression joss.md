# incremental_linear Incremental leaning for Bayesian linear regression

Song Congwei1

1Yanqi Lake Beijing Institute of Mathematical Sciences and Applications  101408

**Abstract** Industrial activity is always accumulating data. According to the ordinary machine learning algorithm, it is not efficient to process all the data at once. Incremental learning is perfect for this situation....



**Keywords** I incremental learning；Bayesian linear regression；dyeing industry；evidence approximation procedure；marginal maxi-mum likelihood



## Summary

Data in high-dim accumulated contiuously in the industry. Common ML becomes inefficient to handle with the large data.

`incremental_linear` is a Python library to implement incremental learning based on Bayesian linear regression.

## Statement of Need

SGD is employed to do the work. But it dose not converge necessarily. and would slows down when working with large data. Traditional linear regressions including baysian ridge regression implemented by scikit-learn are not designed for incremental learning.

## Principle



### Bayesian method for incremental learning

$$
p(\theta |D, D') =\frac{p(D'|\theta)}{\int p(\theta|D)p(D'|\theta)\mathrm{d}\theta}p(\theta|D)
$$
where $D$ is the learned data，$D'$ is the new data。post distr. $p(\theta |D)$，now is regard as priori distr in (1)。

In Bayesian linear model, we have 
$$
y=\Phi w + \epsilon, \epsilon\sim N(0,\sigma^2)
$$

where $\Phi$ is a certain design matrix, and assume
$$
w_i\sim N(0,\alpha_i^{-1})
$$
where $\sigma,\alpha_i$ are the hyper-parameters should be fitted.



According to flat distribution of the hyper-parameters, we get $w\sim N(\mu,\Sigma)$，where
$$
\mu = \sigma^{-2}\Sigma \Phi^T y= \sigma^{-2}\Sigma b\\
\Sigma =(A+\sigma^{-2}\Phi^T\Phi)^{-1}= (A+\sigma^{-2}G)^{-1}, A=\mathrm{diag}(\alpha)
$$
$\mu$ is the estimator of $w$. confidence is determined by $\sigma^2+\phi^T(x')\Sigma\phi(x')$, where $\phi(x)=(\phi_1(x'),\cdots,\phi_p(x'))^T$.

The hyper-parameters $\alpha,\sigma^2$ are computered by EAP。



### Algorithm

Now we have $N$ samples $(x_i,y_i),i=1,\cdots,N$ named data $D$

Assign variables $b=\Phi^Ty,G=\Phi^T\Phi,q=\|{y}\|^2$, or $b=\frac{\Phi^Ty}{N},G=\frac{\Phi^T\Phi}{N},q=\frac{\|{y}\|^2}{N}$，if large amount of data would make overflow error.


In most cases, $N$ is large enough to make the flowing approximation
$$
\sigma^{2}\sim \mu^TG\mu-2\mu^T b+q
$$



After execute EAP, we obtain the posterior distr. of $w|D\sim N(\mu,\Sigma)$ under data $D$, which will be the prior distr for the incremental data $D'=\{(x_i,y_i)\}$. Corespondingly, let $b'=\Phi'^Ty',G'=\Phi'^T\Phi',q=\|{y'}\|^2$.

Again we get the posterior distr. $w|D'\sim N(\mu',\Sigma')$，where
$$
\mu' = \Sigma'(\sigma^{-2}b'+\Sigma^{-1}\mu)=(\sigma^2+\Sigma G')^{-1}(\Sigma b'+\sigma^2\mu)\\
\Sigma' =(\Sigma^{-1}+\sigma^{-2}G')^{-1}=\sigma^2(\sigma^2+\Sigma G')^{-1}\Sigma
$$
Meanwhile $\sigma^2$ is updated as
$$
\sigma^{2}\leftarrow (1-r)\sigma^{2}+r\sigma'^2,
$$
where $\sigma'^2=\mu'^TG’\mu'-2\mu’^T b’+q'$ and $r=\frac{N'}{N+N'}$. Since (7) or (5) is an approximating formula, we need not stubbornly insist on the definition $r=\frac{N'}{N+N'}$. It is better to regard $r$ as the weight of the new data that could be tuned dependently. Let $r=1$, for transfer learning.

**Algorithm**

train model with old data by EAP and store $\sigma,\mu,\Sigma$

Input new data $(X',Y')$

assign varaibles $G\leftarrow \Phi'^T\Phi', b\leftarrow \Phi'^Ty',q\leftarrow \|y'\|^2$

update in single step：

1. update $\mu,\Sigma$ with (12)
2. update $\sigma^2$ with (13)

return $\sigma^2$



## Experiment

### Toy experiment

```python
print('receive data')
X=np.array([[1,2,1],[3,3,2], [4,5,3],[5,6,4]])
y=np.array([3, 6,9.5,10.5])
print('create a model (warm_start=True by default)')
ilr = IncrementalLinearRegression(warm_start=True)
ilr.fit(X, y)
print(f'''
coef: {ilr.coef_}
training score: {ilr.score(X, y)}
''')

print('Saving the model')
import joblib
joblib.dump(ilr, 'ilr.model')
print('Saved the model')

print('Loading the model')
ilr = joblib.load('ilr.model')

print('Receive new data')
print(f'''previous coef: {ilr.coef_}
make sure that warm_start is set to be True
    ''')
X = np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
y = np.array([11, 8,11,13])
ilr.fit(X, y)
print(f'''After incremental learning.
coef: {ilr.coef_}
training score: {ilr.score(X, y)}
informative features: {ilr.informative_features(0.001)}
''')
```

In the exp，we devide the data to three parts. 分别代表旧训练数据、新训练数据、测试数据。

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



实验证明本问算法的有效性，也证实了增量学习会是未来工业处理大数据的主要机器学习范式。



## Acknowledgements





**参考文献**

杨晓伟 郝志峰 支持向量机的算法设计与分析【M】北京：科学出版社2013: 143-147。Yang X Hao Z Algorithm Design and Analysis of SVM Beijing Science Publisher 2013: 143-147

Walter G., Augustin T. (2010) Bayesian Linear Regression — Different Conjugate Models and Their (In)Sensitivity to Prior-Data Conflict[M]// Kneib T., Tutz G. Statistical Modelling and Regression Structures. Physica-Verlag HD. 2009: 59-78

Fletcher T. Relevance Vector Machines Explained 2010. http://home.mit.bme.hu/~horvath/IDA/RVM.pdf

Hastie T, Trevor R, Tibshirani J. THE ELEMENTS OF STATISTICAL LEARNING: DATA MINING, INFERENCE, AND PREDICTION, SECOND EDITION[M]. Springer, 2001.44-49, 272-279



M. Sugiyama, Introduction to Statistical Machine Learning  Elsevier (Singapore),2016. 94-95, 162-166.



M E Tipping Sparse Bayesian Learning and the Relevance Vector Machine[J]. Journal of Machine Learning Research, 2001, 1: 211-244.



Tzikas D., Likas A., Galatsanos N. (2008) Incremental Relevance Vector Machine with Kernel Learning[C]. In: Darzentas J., Vouros G.A., Vosinakis S., Arnellos A. (eds) Artificial Intelligence: Theories, Models and Applications. SETN 2008. Lecture Notes in Computer Science, vol 5138. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-87881-0_27



Laskov P Gehl C, Kruger S, Muller K R. Incremental support Vector Machine learning: analysis, implementation and application[J]. Journal of Machine Learning Research, 2006, 7:1909-1936.
