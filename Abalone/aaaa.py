# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 19:00:48 2018

@author: wulei

(x-min)/max
train: 0.82
test: 0.730487
train: 0.73
test: 0.7438292
train: 0.82
test: 0.7418279
train: 0.78
test: 0.7344897




"""
import tensorflow as tf
from Data2 import matBatchData
from random import shuffle
import pickle

DataDir = 'E:/Python/matData/'

'''
X:数据
y:标签
cv:k折交叉验证中的k参数
'''
def cross_val_score(X,y,cv=3):
    #所有数据个数
    nb_data = len(X)
    data,label = cross_split(X,y,cv,nb_data)
    #测试正确的样本个数
    nb_correct = 0
    for d,l in zip(data,label):
        nb_correct +=compute_correct_simple(d,l)
    
    return nb_correct/nb_data
    
def cross_split(X,y,cv,nb_data):
    start = 0
    data = []
    label = []
    #将数据分为k份时每份数据的个数
    nb_batch = nb_data//cv
    #将数据打乱
    index = [j for j in range(nb_data)]
    shuffle(index)
    X = X[index]
    y = y[index]
    #将数据划分
    for i in range(cv):
        end = start + nb_batch
        if i == cv-1:
            data.append(X[start:])
            label.append(y[start:])
        else:
            data.append(X[start:end])
            label.append(y[start:end])
            start = end
    return data,label

def loss_func2(label,pred):
    l = 5
    part1 = tf.log(tf.reduce_sum(tf.exp(pred*l)*label,1)+1e-10)/l
    part2 = tf.log(tf.reduce_sum(tf.exp(pred),1)+1e-10)
    return tf.reduce_mean(part2-part1)
    
def loss_func(label,pred):
    #根据label中1的个数n,fx=(f(x1)+f(x2)+...+f(xn))/n
    #label=[0,1/n,1/n,0,0,.....,0,1/n ]
    fx = tf.reduce_sum(label*pred,1,keepdims=True)
    #pred-fx
    pred_ = tf.reduce_sum(tf.exp(tf.subtract(pred,fx)),1)
    loss = -tf.reduce_mean(tf.log(tf.clip_by_value(1/pred_,0.001,1)))
    return loss
    
def loss_func1(label,pred):
    batch_array = (1 - label) + pred*label
    batch_loss = tf.pow(tf.reduce_prod(batch_array,1),1/tf.reduce_sum(label,1))
    loss = -tf.reduce_mean(tf.log(batch_loss))
    return loss

def compute_correct_simple(v_xs, v_ys):
    y_pre = sess.run(output, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    nb_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return nb_correct

def compute_accuracy(v_xs, v_ys):
    global output
    y_pre = sess.run(output, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def weight_variable(shape,init,stddev,name):
    weight = tf.Variable(init(shape,stddev=stddev),name=name)
    return weight
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def add_to_pickle(dd):
    f = open('output.txt','wb')
    pickle.dump(dd,f)
    f.close()
    
dd = {}
d1 = {}
d2 = {}
d3 = {}
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 38])   # 28x28
ys = tf.placeholder(tf.float32, [None, 13])

#weight1 = weight_variable([38,20],
#                         tf.truncated_normal,0.1,'w1')
##偏置
#bias1 = bias_variable([20])
#
#layer1 = tf.nn.tanh(tf.matmul(xs,weight1) + bias1)

#权重
weight1 = weight_variable([38,20],
                         tf.truncated_normal,0.1,'w1')

#tf.add_to_collection('loss',tf.contrib.layers.l2_regularizer(0.0001)(weight1))
#偏置
bias1 = bias_variable([20])

layer1 = tf.nn.tanh(tf.matmul(xs,weight1) + bias1)
#
##权重
#weight2 = weight_variable([400,200],
#                         tf.truncated_normal,0.1,'w2')
##tf.add_to_collection('loss',tf.contrib.layers.l2_regularizer(0.0001)(weight2))
##偏置
#bias2 = bias_variable([200])
#
#layer2 = tf.nn.tanh(tf.matmul(layer1,weight2) + bias2)

##权重
#weight4 = weight_variable([200,100],
#                         tf.truncated_normal,0.1,'w4')
##偏置
#bias4 = bias_variable([100])
#
#layer4 = tf.nn.tanh(tf.matmul(layer2,weight4) + bias4)

#权重
weight3 = weight_variable([20,13],
                         tf.truncated_normal,0.1,'w3')
#tf.add_to_collection('loss',tf.contrib.layers.l2_regularizer(0.0001)(weight3))
#偏置
bias3 = bias_variable([13])
#计算输出
output = tf.matmul(layer1,weight3) + bias3
#使用激活函数
#output = tf.nn.softmax(output)

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(output,0.001,1)),
#                                              reduction_indices=[1]))       # loss
#cross_entropy = loss_func(ys,output)
#cross_entropy = loss_func1(ys,tf.clip_by_value(output,0.001,1)) #loss_func1
#l2_loss = tf.add_n(tf.get_collection('loss'))
cross_entropy = loss_func2(ys,output) 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

data = matBatchData(DataDir+'BirdSong.mat')
data.splitData(0.3)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
a = 0
for i in range(10000):
    batch_xs, batch_ys, batch_ys_= data.next_batch(100)
#    batch_ys_ = matBatchData.deal_data1(batch_ys_)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys_})
#    print(sess.run(output,feed_dict={xs: batch_xs}))
#    print(batch_xs)
    if i % 50 == 0:
#        print("test:",compute_accuracy(
#            data.data_test, data.target_test))
        print(i/50)
        print("train:",compute_accuracy(batch_xs,batch_ys))
#        output_ = sess.run(output,feed_dict={xs: batch_xs})
#        w1 = sess.run(weight1)
#        w2 = sess.run(weight2)
#        w3 = sess.run(weight3)
#        dd[i/50] = output_
#        d1[i/50] = w1
#        d2[i/50] = w2
#        d3[i/50] = w3
#        add_to_pickle(dd)
        print("test:",compute_accuracy(
            data.data_test, data.target_test))
        print('loss:',sess.run(cross_entropy,feed_dict={xs:batch_xs, ys:batch_ys_}))


