# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:48:51 2019

@author: Administrator
"""

#import tensorflow as tf
#import numpy as np
#
#ys = np.array([[0,1,1,0,1],[1,1,0,0,1]])
#
#layer1 = np.array([[0.5,0.2,0.1,0.1,0.1],[0.3,0.2,0.1,0.1,0.3]])
#
##a1 = tf.clip_by_value(layer1,0.001,1)
##a2 = ys*a1
##a3 = tf.reduce_sum(a2,reduction_indices=[1])
##a4 = tf.log(a3)
#
#
##cross_entropy = -tf.reduce_mean(a4)
#
##cross_entropy = -tf.reduce_mean(tf.log(tf.reduce_sum(ys*tf.clip_by_value(output,0.001,1),reduction_indices=[1])))
#cross_entropy = -tf.reduce_mean(tf.log(tf.reduce_sum(ys*tf.clip_by_value(layer1,0.001,1),reduction_indices=[1])))

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

AbaloneData = imp.load_source('Abalone','../Data.py')



parser = argparse.ArgumentParser(description="show some parse for the progess")
parser.add_argument("--DataDir",type=str,default="../uci_data_set/Abalone.mat",help="the mat data file path")
parser.add_argument("--train_epoch",type=int,default=5000,help="number of training epochs")
parser.add_argument("--model_save_name",type=str,default="Abalone.ckpt",help="the file name for model to save")
parser.add_argument("--model_save_dir",type=str,default=None,help="the dir for model to save")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
parser.add_argument("--use_gpu", default=True, help="Whether to use GPU")
#parser.add_argument("--nb_item", type=int, required=True, choices=[0,1,2,3,4], help="crosssplit data ,it select which group to use")
parser.add_argument("--reload_data", default=False, help="Whether to reload data")
parse = parser.parse_args()

if parse.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def loss_func2(label,pred):
    l = 5
    part1 = tf.log(tf.reduce_sum(tf.exp(pred*l)*label,1)+1e-10)/l
    part2 = tf.log(tf.reduce_sum(tf.exp(pred),1)+1e-10)
    return tf.reduce_mean(part2-part1)

def loss_func(label,pred):
    #根据label中1的个数n,fx=(f(x1)+f(x2)+...+f(xn))/n
    #label=[0,1/n,1/n,0,0,.....,0,1/n ]
    fx = tf.reduce_sum(label*pred,1,keep_dims=True)
    #pred-fx
    pred_ = tf.reduce_sum(tf.exp(tf.subtract(pred,fx)),1)
    loss = -tf.reduce_mean(tf.log(tf.clip_by_value(1/pred_,0.001,1)))
    return loss

def loss_func1(label,pred):
    batch_array = (1 - label) + pred*label
    batch_loss = tf.pow(tf.reduce_prod(batch_array,1),1/tf.reduce_sum(label,1))
    loss = -tf.reduce_mean(tf.log(batch_loss))
    return loss

def compute_accuracy(xs_g,ys_g,v_xs, v_ys,output,sess):
    y_pre = sess.run(output, feed_dict={xs_g: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs_g: v_xs, ys_g: v_ys})
    return result
    
def compute_correct_number(xs_g,ys_g,v_xs, v_ys,output,sess):
    y_pre = sess.run(output, feed_dict={xs_g: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    correct_number = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    num = sess.run(correct_number, feed_dict={xs_g: v_xs, ys_g: v_ys})
    return num

def weight_variable(shape,init,stddev,name):
    weight = tf.Variable(init(shape,stddev=stddev),name=name)
    return weight

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape,name=name)
    return tf.Variable(initial)

#定义权重偏置
def define_WB(NetworkShape):
    Weigth = []
    Bias = []
    for n in range(len(NetworkShape)-1):
        n_in = NetworkShape[n]
        n_out = NetworkShape[n+1]
        W = weight_variable([n_in,n_out],tf.truncated_normal,0.1,'w'+str(n))
        B = bias_variable([n_out],'b'+str(n))
        Weigth.append(W)
        Bias.append(B)
    return Weigth,Bias

def infer(xs,W,B):
    layer1 = tf.nn.softmax(tf.matmul(xs,W[0]) + B[0])
    return layer1

def train(output,ys):
    cross_entropy = loss_func2(ys,output) 
    train_step = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy)
    return train_step,cross_entropy

def getGraph(NetworkShape,cv=5):
    createVar = locals()
    for i in range(5):
        createVar['g'+str(i)] = tf.Graph()
        createVar['xs_g'+str(i)] = tf.placeholder(tf.float32, [None, 7])
        createVar['ys_g'+str(i)] = tf.placeholder(tf.float32, [None, 29])
        

# define placeholder for inputs to network
#xs = tf.placeholder(tf.float32, [None, 7])   # 28x28
#ys = tf.placeholder(tf.float32, [None, 29])
#keep_prob = tf.placeholder(tf.float32)

g1 = tf.Graph()
with g1.as_default():
    with tf.variable_scope("Graph1"):
        xs_g1 = tf.placeholder(tf.float32, [None, 7])   # 28x28
        ys_g1 = tf.placeholder(tf.float32, [None, 29])
        weight1 = weight_variable([7,29],
                                 tf.truncated_normal,0.1,'w1')
        #偏置
        bias1 = bias_variable([29],'b1')
        
        output1 = tf.nn.softmax(tf.matmul(xs_g1,weight1) + bias1)
        
        cross_entropy1 = loss_func2(ys_g1,output1)
        train_step1 = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy1)
    
g2 = tf.Graph()
with g2.as_default():
    with tf.variable_scope("Graph2"):
        xs_g2 = tf.placeholder(tf.float32, [None, 7])   # 28x28
        ys_g2 = tf.placeholder(tf.float32, [None, 29])
        weight2 = weight_variable([7,29],
                                 tf.truncated_normal,0.1,'w1')
        #偏置
        bias2 = bias_variable([29],'b1')
        
        output2 = tf.nn.softmax(tf.matmul(xs_g2,weight2) + bias2)
        
        cross_entropy2 = loss_func2(ys_g2,output2)
        train_step2 = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy2)

g3 = tf.Graph()
with g3.as_default():
    with tf.variable_scope("Graph3"):
        xs_g3 = tf.placeholder(tf.float32, [None, 7])   # 28x28
        ys_g3 = tf.placeholder(tf.float32, [None, 29])
        weight3 = weight_variable([7,29],
                                 tf.truncated_normal,0.1,'w1')
        #偏置
        bias3 = bias_variable([29],'b1')
        
        output3 = tf.nn.softmax(tf.matmul(xs_g3,weight3) + bias3)
        
        cross_entropy3 = loss_func2(ys_g3,output3)
        train_step3 = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy3)
    
g4 = tf.Graph()
with g4.as_default():
    with tf.variable_scope("Graph4"):
        xs_g4 = tf.placeholder(tf.float32, [None, 7])   # 28x28
        ys_g4 = tf.placeholder(tf.float32, [None, 29])
        weight4 = weight_variable([7,29],
                                 tf.truncated_normal,0.1,'w1')
        #偏置
        bias4 = bias_variable([29],'b1')
        
        output4 = tf.nn.softmax(tf.matmul(xs_g4,weight4) + bias4)
        
        cross_entropy4 = loss_func2(ys_g4,output4)
        train_step4 = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy4)
    
g5 = tf.Graph()
with g5.as_default():
    with tf.variable_scope("Graph5"):
        xs_g5 = tf.placeholder(tf.float32, [None, 7])   # 28x28
        ys_g5 = tf.placeholder(tf.float32, [None, 29])
        weight5 = weight_variable([7,29],
                                 tf.truncated_normal,0.1,'w1')
        #偏置
        bias5 = bias_variable([29],'b1')
        
        output5 = tf.nn.softmax(tf.matmul(xs_g5,weight5) + bias5)
        
        cross_entropy5 = loss_func2(ys_g5,output5)
        train_step5 = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy5)


Abalone_data = AbaloneData.UCIData(parse.DataDir)
X,Y,partial_Y = Abalone_data.createPartialData()
item_data = CrossData(X)
item_data.cross_split()

correct_pred = []
with tf.Session(graph=g1) as sess1:
    init = tf.global_variables_initializer()
    sess1.run(init)
    for i in range(parse.train_epoch):
        batch_index= item_data.next_batch(parse.batch_size,0)
        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
        sess1.run(train_step1, feed_dict={xs_g1: batch_xs, ys_g1: batch_ys_})
        if i % 50 == 0:
            print('training 1:',i/50)
            print("train:",compute_accuracy(xs_g1,ys_g1,batch_xs,batch_ys,output1,sess1))
            print("test:",compute_accuracy(xs_g1,ys_g1,
                X[item_data.data_test[0]], Y[item_data.data_test[0]], output1,sess1))
            print('loss:',sess1.run(cross_entropy1,feed_dict={xs_g1:batch_xs, ys_g1:batch_ys_}))
    corr_num = compute_correct_number(xs_g1,ys_g1,X[item_data.data_test[0]], Y[item_data.data_test[0]], output1,sess1)
    correct_pred.append(corr_num)
    item_data.set_index_to_zero()

with tf.Session(graph=g2) as sess2:
    init = tf.global_variables_initializer()
    sess2.run(init)
    for i in range(parse.train_epoch):
        batch_index= item_data.next_batch(parse.batch_size,1)
        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
        sess2.run(train_step2, feed_dict={xs_g2: batch_xs, ys_g2: batch_ys_})
        if i % 50 == 0:
            print('training 1:',i/50)
            print("train:",compute_accuracy(xs_g2,ys_g2,batch_xs,batch_ys,output2,sess2))
            print("test:",compute_accuracy(xs_g2,ys_g2,
                X[item_data.data_test[1]], Y[item_data.data_test[1]], output2,sess2))
            print('loss:',sess2.run(cross_entropy2,feed_dict={xs_g2:batch_xs, ys_g2:batch_ys_}))
    corr_num = compute_correct_number(xs_g2,ys_g2,X[item_data.data_test[1]], Y[item_data.data_test[1]], output2,sess2)
    correct_pred.append(corr_num)
    item_data.set_index_to_zero()
    
with tf.Session(graph=g3) as sess3:
    init = tf.global_variables_initializer()
    sess3.run(init)
    for i in range(parse.train_epoch):
        batch_index= item_data.next_batch(parse.batch_size,2)
        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
        sess3.run(train_step3, feed_dict={xs_g3: batch_xs, ys_g3: batch_ys_})
        if i % 50 == 0:
            print('training 1:',i/50)
            print("train:",compute_accuracy(xs_g3,ys_g3,batch_xs,batch_ys,output3,sess3))
            print("test:",compute_accuracy(xs_g3,ys_g3,
                X[item_data.data_test[2]], Y[item_data.data_test[2]], output3,sess3))
            print('loss:',sess3.run(cross_entropy3,feed_dict={xs_g3:batch_xs, ys_g3:batch_ys_}))
    corr_num = compute_correct_number(xs_g3,ys_g3,X[item_data.data_test[2]], Y[item_data.data_test[2]], output3,sess3)
    correct_pred.append(corr_num)
    item_data.set_index_to_zero()
    
with tf.Session(graph=g4) as sess4:
    init = tf.global_variables_initializer()
    sess4.run(init)
    for i in range(parse.train_epoch):
        batch_index= item_data.next_batch(parse.batch_size,3)
        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
        sess4.run(train_step4, feed_dict={xs_g4: batch_xs, ys_g4: batch_ys_})
        if i % 50 == 0:
            print('training 1:',i/50)
            print("train:",compute_accuracy(xs_g4,ys_g4,batch_xs,batch_ys,output4,sess4))
            print("test:",compute_accuracy(xs_g4,ys_g4,
                X[item_data.data_test[3]], Y[item_data.data_test[3]], output4,sess4))
            print('loss:',sess4.run(cross_entropy4,feed_dict={xs_g4:batch_xs, ys_g4:batch_ys_}))
    corr_num = compute_correct_number(xs_g4,ys_g4,X[item_data.data_test[3]], Y[item_data.data_test[3]], output4,sess4)
    correct_pred.append(corr_num)
    item_data.set_index_to_zero()
    
with tf.Session(graph=g5) as sess5:
    init = tf.global_variables_initializer()
    sess5.run(init)
    for i in range(parse.train_epoch):
        batch_index= item_data.next_batch(parse.batch_size,4)
        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
        sess5.run(train_step5, feed_dict={xs_g5: batch_xs, ys_g5: batch_ys_})
        if i % 50 == 0:
            print('training 1:',i/50)
            print("train:",compute_accuracy(xs_g5,ys_g5,batch_xs,batch_ys,output5,sess5))
            print("test:",compute_accuracy(xs_g5,ys_g5,
                X[item_data.data_test[4]], Y[item_data.data_test[4]], output5,sess5))
            print('loss:',sess5.run(cross_entropy5,feed_dict={xs_g5:batch_xs, ys_g5:batch_ys_}))
    corr_num = compute_correct_number(xs_g5,ys_g5,X[item_data.data_test[4]], Y[item_data.data_test[4]], output5,sess5)
    correct_pred.append(corr_num)


#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    for i in range(parse.train_epoch):
#        batch_index= item_data.next_batch(parse.batch_size,0)
#        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
#    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
#        sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys_})
#        if i % 50 == 0:
#            print('training 1:',i/50)
#            print("train:",compute_accuracy(batch_xs,batch_ys,output1))
#            print("test:",compute_accuracy(
#                X[item_data.data_test[0]], Y[item_data.data_test[0]], output1))
#            print('loss:',sess.run(cross_entropy1,feed_dict={xs:batch_xs, ys:batch_ys_}))
#    corr_num = compute_correct_number(X[item_data.data_test[0]], Y[item_data.data_test[0]], output1)
##    print('corr_num:',corr_num)
#    correct_pred.append(corr_num)
#    item_data.set_index_to_zero()
#
#
#    for i in range(parse.train_epoch):
#        batch_index= item_data.next_batch(parse.batch_size,1)
#        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
#    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
#        sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys_})
#        if i % 50 == 0:
#            print('training 2:',i/50)
#            print("train:",compute_accuracy(batch_xs,batch_ys,output2))
#            print("test:",compute_accuracy(
#                X[item_data.data_test[1]], Y[item_data.data_test[1]], output2))
#            print('loss:',sess.run(cross_entropy1,feed_dict={xs:batch_xs, ys:batch_ys_}))
#    corr_num = compute_correct_number(X[item_data.data_test[1]], Y[item_data.data_test[1]], output2)
#    correct_pred.append(corr_num)
#    item_data.set_index_to_zero()
#
#    for i in range(parse.train_epoch):
#        batch_index= item_data.next_batch(parse.batch_size,2)
#        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
#    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
#        sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys_})
#        if i % 50 == 0:
#            print('training 3:',i/50)
#            print("train:",compute_accuracy(batch_xs,batch_ys,output3))
#            print("test:",compute_accuracy(
#                X[item_data.data_test[2]], Y[item_data.data_test[2]], output3))
#            print('loss:',sess.run(cross_entropy1,feed_dict={xs:batch_xs, ys:batch_ys_}))
#    corr_num = compute_correct_number(X[item_data.data_test[2]], Y[item_data.data_test[2]], output3)
#    correct_pred.append(corr_num)
#    item_data.set_index_to_zero()
#
#    for i in range(parse.train_epoch):
#        batch_index= item_data.next_batch(parse.batch_size,3)
#        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
#    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
#        sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys_})
#        if i % 50 == 0:
#            print('training 4:',i/50)
#            print("train:",compute_accuracy(batch_xs,batch_ys,output4))
#            print("test:",compute_accuracy(
#                X[item_data.data_test[3]], Y[item_data.data_test[3]], output4))
#            print('loss:',sess.run(cross_entropy1,feed_dict={xs:batch_xs, ys:batch_ys_}))
#    corr_num = compute_correct_number(X[item_data.data_test[3]], Y[item_data.data_test[3]], output4)
#    correct_pred.append(corr_num)
#    item_data.set_index_to_zero()
#    
#    for i in range(parse.train_epoch):
#        batch_index= item_data.next_batch(parse.batch_size,4)
#        batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
#    #    batch_ys_ = matBatchData.deal_data1(batch_ys_)
#        sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys_})
#        if i % 50 == 0:
#            print('training 5:',i/50)
#            print("train:",compute_accuracy(batch_xs,batch_ys,output5))
#            print("test:",compute_accuracy(
#                X[item_data.data_test[4]], Y[item_data.data_test[4]], output5))
#            print('loss:',sess.run(cross_entropy1,feed_dict={xs:batch_xs, ys:batch_ys_}))
#    corr_num = compute_correct_number(X[item_data.data_test[4]], Y[item_data.data_test[4]], output5)
#    correct_pred.append(corr_num)
#    item_data.set_index_to_zero()
#
Accu = sum(correct_pred)/item_data._num_examples
print('5CrossAccu:',Accu)



