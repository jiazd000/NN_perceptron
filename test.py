import numpy as np

ss=np.ones(1)
x=np.array([1.1,1.0])
# x=[1,1]
# x.insert(0,1.0)
qq=np.hstack((ss,x))
print ss,qq
w=np.array([2,1])

print np.dot(x,w)