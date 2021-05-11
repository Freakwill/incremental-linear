# incremental-linear
incremental bayesian linear model


## Example

```python
X=np.array([[1,2,1],[3,3,2], [4,5,3],[5,6,4]])
y=np.array([3, 6,9.5,10.5])
# you have to set warm_start=True, otherwise, the parameters are reset when fitting next time
a = IncrementalLinearRegression(warm_start=True)
a.fit(X, y)
print(a.predict(X), a.score(X, y))
X=np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
y=np.array([11, 8,11,13])
a.fit(X, y)
print(a.predict(X), a.score(X, y))
print(a.alpha)
print(a.important_features(0.001))
```
