# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:41:33 2019

@author: Administrator
"""
# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
# catch exception with: except Exception as e
#from builtins import range
# End: Python 2/3 compatability header small
import numpy as np
from MyDataDealer import matData,CrossData

class UCIData(matData):
    def __init__(self,dataFileName):
        super(UCIData,self).__init__(dataFileName)
        self.data,self.target = self.loadmatData()
    
    def CreatPLExamplePQ(self,rate,nb_partial_target):
        '''
        该函数生成样本的新标签集,使rate(百分比),nb_partial_target(1<=n<=3)个候选标签
        target:样本的真实标签
        rate:0-1间的一个数,代表取多少的数据添加候选标签
        nb_partial_target:添加多少个候选标签
        '''
        self.partial_target = self.target.copy()
        #数据总个数
        nb_target = len(self.partial_target)
        #用于添加候选标签数据的个数
        nb_partial = int(nb_target*rate)
        #取出需要添加候选标签的数据
        partial_data_temp = self.partial_target[:nb_partial]
        
        for i in partial_data_temp:
            #找出label中数值为0的索引
            zero_index = np.where(i==0)
            
            #随机从zero_index中选择几个位置,把位置保存到one_index
            one_index = set()
            while True:
                if len(one_index) == nb_partial_target:
                    break
                r = np.random.randint(len(zero_index[0]))
                one_index.add(zero_index[0][r])
            #把对应的index的数字改为1
            for index in one_index:
                i[index] = 1
        
        self.partial_target[:nb_partial] = partial_data_temp
        return self.partial_target
    
    #创建偏标记数据
    def createPartialData(self,rate=0.5,nb_partial_target=1):
        if not isinstance(type(self.data),np.ndarray) and isinstance(type(self.target),np.ndarray):
            raise TypeError('data type must be numpy.ndarray')
        nb_data = len(self.data)
        #打乱数据
        index = [i for i in range(nb_data)]
        np.random.shuffle(index)
        self.data = self.data[index]
        self.target = self.target[index]
        #创建偏标记数据
        self.CreatPLExamplePQ(rate,nb_partial_target)
        
        return (self.data,self.target,self.partial_target)
    
    

if __name__=="__main__":
    a = UCIData('./uci_data_set/Abalone.mat')
    a0,a1,a2 = a.createPartialData()
    c = CrossData(a0)
    c.cross_split()
#    c = NextBatchData(a0)
    l = []
    for i in range(40):
        ind = c.next_batch(100,0)
        print(type(ind),len(ind))
        l.append(ind)










