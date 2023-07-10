import numpy as np

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=1), 1)
    return a

Query = np.array([
    [1,0,2],
    [2,2,2],
    [2,1,3]
])

Key = np.array([
    [0,1,1],
    [4,4,0],
    [2,3,1]
])

Value = np.array([
    [1,2,3],
    [2,8,0],
    [2,6,3]
])

scores = Query @ Key.T
print(scores)
# [[ 2  4  4]
#  [ 4 16 12]
#  [ 4 12 10]]

scores = soft_max(scores)
print(scores)
# [[6.33789383e-02 4.68310531e-01 4.68310531e-01]
#  [6.03366485e-06 9.82007865e-01 1.79861014e-02]
#  [2.95387223e-04 8.80536902e-01 1.19167711e-01]]

out = scores @ Value
print(out)
# [[1.93662106 6.68310531 1.59506841]
#  [1.99999397 7.9639916  0.05397641]
#  [1.99970461 7.75989225 0.35838929]
