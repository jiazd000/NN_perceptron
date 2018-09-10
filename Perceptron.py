# -*- coding: utf-8 -*-
import numpy as np
class Perceptron(object):
    def __init__(self, activator,input_num):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.input_num=input_num

        self.w = np.array([0.0 for _ in range(self.input_num+1)])

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%f\n' % (self.w)
    
    def predict(self, X):
        
        X=np.hstack((np.ones(1),X))
        ss=np.array([np.dot(X,self.w)])
        yy=self.activator(ss)

        return yy

    def train(self, X, y, iteration, rate):
                
        m=X.shape[0]

        X=np.column_stack((np.ones((m,1)),X))
        
        for _ in range(iteration):
            J=self.activator(np.dot(X,self.w))-y
            # print J
            for i in range(self.input_num+1): 
                self.w[i]=self.w[i]-((J*X[:,i]).sum())*rate/m

