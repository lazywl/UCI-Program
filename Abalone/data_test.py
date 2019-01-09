# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:18:42 2019

@author: Administrator
"""
# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
# End: Python 2/3 compatability header small

import tensorflow as tf
import argparse
import imp
import os
import pickle
from MyDataDealer import CrossData,Data

#AbaloneData = imp.load_source('Abalone','../Data.py')



parser = argparse.ArgumentParser(description="show some parse for the progess")
parser.add_argument("--DataDir",type=str,default="../uci data set/Abalone.mat",help="the mat data file path")
parser.add_argument("--train_epoch",type=int,default=50,help="number of training epochs")
parser.add_argument("--model_save_name",type=str,default="Abalone.ckpt",help="the file name for model to save")
parser.add_argument("--model_save_dir",type=str,default=None,help="the dir for model to save")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
parser.add_argument("--use_gpu", default=True, help="Whether to use GPU")
parser.add_argument("--nb_item",default=0,choices=[0,1,2,3,4], help="crosssplit data ,it select which group to use")
parser.add_argument("--reload_data", default=True, help="Whether to reload data")
parse = parser.parse_args()

if parse.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


#Abalone_data = AbaloneData.UCIData(parse.DataDir)
#X,Y,partial_Y = Abalone_data.createPartialData()
#item_data = CrossData(X)
#item_data.cross_split()
#
#l=[]
#for i in range(parse.train_epoch):
#    ind = item_data.next_batch(parse.batch_size,parse.nb_item)
#    print(type(ind),len(ind))
#    l.append(ind)

createVar = globals()
for i in range(5):
    createVar["g"+str(i)] = tf.Graph()
    print(type(createVar["g"+str(i)]))
#    with exec("g{}.as_default()".format(i)):
#        with tf.variable_scope("Graph{}".format(i)):
#            createVar["xs_g"+str(i)] = tf.placeholder(tf.float32, [None, 7])
#            createVar["ys_g"+str(i)] = tf.placeholder(tf.float32, [None, 29])

