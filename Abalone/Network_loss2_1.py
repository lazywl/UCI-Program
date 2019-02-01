# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:01:39 2019

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
from MyDataDealer import NextBatchData,Data

AbaloneData = imp.load_source('Abalone','../Data.py')



parser = argparse.ArgumentParser(description="show some parse for the progess")
parser.add_argument("--DataDir",type=str,default="../uci data set/Abalone.mat",help="the mat data file path")
parser.add_argument("--train_epoch",type=int,default=5000,help="number of training epochs")
parser.add_argument("--model_save_name",type=str,default="Abalone.ckpt",help="the file name for model to save")
parser.add_argument("--model_save_dir",type=str,default=None,help="the dir for model to save")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
parser.add_argument("--use_gpu", default=True, help="Whether to use GPU")
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

def compute_accuracy(v_xs, v_ys,output):
    y_pre = sess.run(output, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
    
def compute_correct_number(v_xs, v_ys,output):
    y_pre = sess.run(output, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    correct_number = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    num = sess.run(correct_number, feed_dict={xs: v_xs, ys: v_ys})
    return num

def weight_variable(shape,init,stddev,name):
    weight = tf.Variable(init(shape,stddev=stddev),name=name)
    return weight

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape,name=name)
    return tf.Variable(initial)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 7])   # 28x28
ys = tf.placeholder(tf.float32, [None, 29])
keep_prob = tf.placeholder(tf.float32)


weight1 = weight_variable([7,29],
                         tf.truncated_normal,0.1,'w1')
#偏置
bias1 = bias_variable([29],'b1')

layer1 = tf.nn.softmax(tf.matmul(xs,weight1) + bias1)


#weight2 = weight_variable([38,20],
#                         tf.truncated_normal,0.1,'w1')
##偏置
#bias2 = bias_variable([20])
#
#layer2 = tf.nn.tanh(tf.matmul(xs,weight1) + bias1)

cross_entropy = -tf.reduce_mean(tf.log(tf.reduce_sum(ys*tf.clip_by_value(layer1,0.001,1),reduction_indices=[1])))
train_step = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy)

Abalone_data = AbaloneData.UCIData(parse.DataDir)
X,Y,partial_Y = Abalone_data.createPartialData()
train_data,train_target,train_partial_target,test_data,test_target,test_partial_target = Data.dataSplit(X,Y,partial_Y)

item_data = NextBatchData(train_data)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(parse.train_epoch):
        batch_index= item_data.next_batch(parse.batch_size)
        batch_xs, batch_ys, batch_ys_ = train_data[batch_index],train_target[batch_index],train_partial_target[batch_index]
        batch_ys_ = Data.deal_data(batch_ys_)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys_})
        if i % 50 == 0:
            print(i/50)
            print("train:",compute_accuracy(batch_xs,batch_ys,layer1))
            print("test:",compute_accuracy(
                test_data, test_target, layer1))
            print('loss:',sess.run(cross_entropy,feed_dict={xs:batch_xs, ys:batch_ys_}))






