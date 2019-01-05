# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 08:59:30 2019

@author: Administrator
将具有真实标签的数据变为偏标记数据集
"""
# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

from scipy.io import loadmat
import numpy as np

def CreatPLExamplePQ(target,rate,nb_partial_label):
    '''
    该函数生成样本的新标签集,使rate(百分比),nb_partial_label(1<=n<=3)个候选标签
    target:样本的真实标签
    rate:0-1间的一个数,代表取多少的数据添加候选标签
    nb_partial_label:添加多少个候选标签
    '''
    #数据总个数
    nb_target = len(target)
    #用于添加候选标签数据的个数
    nb_partial = int(nb_target*rate)
    #取出需要添加候选标签的数据
    partial_data_temp = target[:nb_partial]
    
    for i in partial_data_temp:
        #找出label中数值为0的索引
        zero_index = np.where(i==0)
        
        #随机从zero_index中选择几个位置,把位置保存到one_index
        one_index = set()
        while True:
            if len(one_index) == nb_partial_label:
                break
            r = np.random.randint(len(zero_index[0]))
            one_index.add(zero_index[0][r])
        #把对应的index的数字改为1
        for index in one_index:
            i[index] = 1
    
    target[:nb_partial] = partial_data_temp
    return target

#创建偏标记数据
def createPartialData(data,target,rate,nb_partial_label):
    if not isinstance(type(data),np.ndarray) and isinstance(type(target),np.ndarray):
        raise TypeError('data type must be numpy.ndarray')
    nb_data = len(data)
    #打乱数据
    index = [i for i in range(nb_data)]
    np.random.shuffle(index)
    data = data[index]
    target = target[index]
    #创建偏标记数据
    partial_target = CreatPLExamplePQ(target.copy(),rate,nb_partial_label)
    
    return (data,target,partial_target)
    

def getData(data_file_path,rate=0.5,nb_partial_label=1):
    d = loadmat(data_file_path)
    data = d['data']
    target = d['target']
    
        
    return createPartialData(data,target,rate,nb_partial_label) 

#a = [[0,0,0,0,0,4,0,0,0],
#     [0,0,0,0,0,0,0,5,0],
#     [0,0,4,0,0,0,0,0,0],
#     [0,0,0,0,0,2,0,0,0],
#     [0,3,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,7]]
#    
#b = CreatPLExamplePQ(np.array(a),0.8,3)
    
    
data,target,partial_target = getData('./uci data set/Abalone.mat',0.6,2)
    
    
    
    
    
    