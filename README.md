# incremental-linear
incremental bayesian linear model


## Example

```python
print('receive data')
X=np.array([[1,2,1],[3,3,2], [4,5,3],[5,6,4]])
y=np.array([3, 6,9.5,10.5])
print('create a model (set warm_start=True)')
a = IncrementalLinearRegression(warm_start=True)
a.fit(X, y)
print(f'''
coef: {a.coef_}
training score: {a.score(X, y)}
''')

print('save the model')
import joblib
joblib.dump(a, 'a.model')
print('load the model')
a= joblib.load('a.model')
print('receive new data')
print(f'''previous coef: {a.coef_}
    flag: {a.flag} (if False, then partial_fit will raise an error!)''')
X=np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
y=np.array([11, 8,11,13])
a.fit(X, y)
print(f'''
coef: {a.coef_}
training score: {a.score(X, y)}
important features: {a.important_features(0.001)}
    ''')
```

OUTPUT:

      receive data
      create a model (set warm_start=True)

          coef: [0.78053444 1.17193856 0.        ]
          training score: 0.9860162630232226

      save the model
      load the model
      receive new data
      previous coef: [0.78053444 1.17193856 0.        ]
              flag: True (if False, then partial_fit will raise an error!)

          coef: [1.30151375 0.80216628 0.        ]
          training score: 0.9772172406517436
          important features: (0, 1)
