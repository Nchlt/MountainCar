import pickle
import numpy as  np


policy = pickle.load(open('bestPol.p', 'rb'))
# policy = pickle.load(open('b2p.p', 'rb'))
# policy = np.floor(policy).reshape(6, 6)

print(type(policy))
print(policy.shape)

print(list(policy))
print(policy)


# np.array([list([0., 1., 0., 0., 0., 1.]), list([0., 0., 1., 2., 0., 0.]), list([0., 0., 0., 0., 2., 0.]), list([1., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 1.]), list([1., 0., 0., 2., 0., 0.])])
# np.list[list([0., 1., 0., 0., 0., 1.]), list([0., 0., 1., 2., 0., 0.]), list([0., 0., 0., 0., 2., 0.]), list([1., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 1.]), list([1., 0., 0., 2., 0., 0.])]
# np.array([list([1., 1., 0., 1., 1., 1.]), list([1., 0., 1., 2., 0., 0.]), list([0., 1., 0., 0., 2., 0.]), list([2., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 2.]), list([1., 0., 1., 2., 0., 0.])])
# print(np.array([list([1., 0., 0., 0., 2., 0.]), list([2., 1., 1., 2., 0., 0.]), list([0., 2., 0., 0., 2., 0.]), list([2., 2., 2., 0., 2., 1.]), list([2., 0., 0., 1., 2., 2.]), list([1., 0., 2., 2., 0., 0.])]))
# print(np.array([list([2., 0., 0., 0., 2., 1.]), list([0., 0., 0., 2., 2., 0.]), list([0., 1., 0., 0., 2., 2.]), list([2., 2., 0., 0., 2., 2.]), list([0., 0., 0., 0., 2., 2.]), list([1., 2., 2., 1., 2., 0.])]))