import cma
import numpy as np
np.random.seed(0)

square = np.vectorize(lambda x : x**2)


X = np.random.randint(low=0, high=3, size=10)
#X = [5 for _ in range(200)]
print(X)

def obj_fun(X):
    X = square(X)
    return X.sum()

print(obj_fun(X))
solution, _ = cma.fmin2(obj_fun, X, 1,{
'BoundaryHandler': 'BoundTransform',
'bounds': [0,3],
'popsize':1000,
'CMA_mu':10,
'ftarget':2
})


solution = np.floor(solution)
print(solution)

# cma.fmin(obj_fun, X, 0.2, {'boundary_handling': 'BoundPenalty',
#                             'bounds':[0,3],
#                             })
