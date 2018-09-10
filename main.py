# -*- coding: utf-8 -*-
import numpy as np
from Perceptron import Perceptron

def f(x):
    '''
    定义激活函数f
    '''
    for i in range(x.shape[0]):
        if x[i]>0:
            x[i]=1.0
        else:
            x[i]=0.0    
    return x


if __name__ == '__main__':

    X = np.array([[1.0,1.0], [0.0,0.0], [1.0,0.0], [0.0,1.0]])
    y = np.array([1.0, 0.0, 0.0, 0.0]) 
    p=Perceptron(f,2)
    p.train(X,y,10,0.1)
    print '1 and 1 = %d' % p.predict(np.array([1.0, 1.0]))
    print '0 and 0 = %d' % p.predict(np.array([0.0, 0.0]))
    print '1 and 0 = %d' % p.predict(np.array([1.0, 0.0]))
    print '0 and 1 = %d' % p.predict(np.array([0.0, 1.0]))
