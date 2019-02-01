# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:35:28 2019

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

def inference(x,W,B,activeFunc_hidden=tf.nn.tanh,activeFunc_output=None):
    layerInput = x
    print('input layer:',W[0].shape.as_list()[0])
    for i in range(len(W)):
        if i == len(W)-1:
            print('output layer:',W[i].shape.as_list()[1])
            if activeFunc_output is None:
                layerOutput = tf.matmul(layerInput,W[i]) + B[i]
                return layerOutput
            layerOutput = activeFunc_output(tf.matmul(layerInput,W[i]) + B[i])
            return layerOutput
        layerOutput = activeFunc_hidden(tf.matmul(layerInput,W[i]) + B[i])
        print('hidden layer%d:'%(i+1),W[i].shape.as_list()[1])
        layerInput = layerOutput
    
def train(output,ys):
    cross_entropy = loss_func2(ys,output) 
    train_step = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy)
    return train_step,cross_entropy

def getGraph(cv=5):
    output_array = []
    cross_entropy_array = []
    train_step_array = []
    Graph_array = []

    createVar = globals()
    for i in range(cv):
        createVar["g"+str(i)] = tf.Graph()
        with createVar["g"+str(i)].as_default():
            with tf.variable_scope("Graph{}".format(i)):
                createVar["xs_g"+str(i)] = tf.placeholder(tf.float32, [None, 7],name='data')
                createVar["ys_g"+str(i)] = tf.placeholder(tf.float32, [None, 29],name='target')
                createVar["weight_g"+str(i)] = weight_variable([7,29],tf.truncated_normal,0.1,'w0')
                createVar["bais_g"+str(i)] = bias_variable([29],'b0')
                createVar["output_g"+str(i)] = tf.nn.softmax(tf.matmul(createVar["xs_g"+str(i)],createVar["weight_g"+str(i)]) + createVar["bais_g"+str(i)])
                createVar["cross_entropy_g"+str(i)] = loss_func2(createVar["ys_g"+str(i)],createVar["output_g"+str(i)])
                createVar["train_step_g"+str(i)] = tf.train.AdamOptimizer(parse.lr).minimize(createVar["cross_entropy_g"+str(i)])
                output_array.append(createVar["output_g"+str(i)])
                cross_entropy_array.append(createVar["cross_entropy_g"+str(i)])
                train_step_array.append(createVar["train_step_g"+str(i)])
        Graph_array.append(createVar["g"+str(i)])
    return output_array,cross_entropy_array,train_step_array,Graph_array



a,b,c,d= getGraph()
a1,b1,c1,d1= getGraph()
a2,b2,c2,d2= getGraph()
a3,b3,c3,d3= getGraph()
a4,b4,c4,d4= getGraph()


## -*- coding: utf-8 -*-
#"""
#Created on Sat Jan  5 20:24:07 2019
#
#@author: Administrator
#"""
## Begin: Python 2/3 compatibility header small
## Get Python 3 functionality:
#from __future__ import\
#    absolute_import, print_function, division, unicode_literals
## End: Python 2/3 compatability header small
#
#import tensorflow as tf
#import argparse
#import imp
#import os
#from MyDataDealer import CrossData,Data
#
#AbaloneData = imp.load_source('Abalone','../Data.py')
#
#
#
#parser = argparse.ArgumentParser(description="show some parse for the progess")
#parser.add_argument("--DataDir",type=str,default="../uci data set/Abalone.mat",help="the mat data file path")
#parser.add_argument("--train_epoch",type=int,default=5000,help="number of training epochs")
#parser.add_argument("--model_save_name",type=str,default="Abalone.ckpt",help="the file name for model to save")
#parser.add_argument("--model_save_dir",type=str,default=None,help="the dir for model to save")
#parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
#parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
#parser.add_argument("--use_gpu", default=True, help="Whether to use GPU")
#parse = parser.parse_args()
#
#if parse.use_gpu:
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#else:
#    os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
#
#def loss_func2(label,pred):
#    l = 5
#    part1 = tf.log(tf.reduce_sum(tf.exp(pred*l)*label,1)+1e-10)/l
#    part2 = tf.log(tf.reduce_sum(tf.exp(pred),1)+1e-10)
#    return tf.reduce_mean(part2-part1)
#
#def loss_func(label,pred):
#    #根据label中1的个数n,fx=(f(x1)+f(x2)+...+f(xn))/n
#    #label=[0,1/n,1/n,0,0,.....,0,1/n ]
#    fx = tf.reduce_sum(label*pred,1,keep_dims=True)
#    #pred-fx
#    pred_ = tf.reduce_sum(tf.exp(tf.subtract(pred,fx)),1)
#    loss = -tf.reduce_mean(tf.log(tf.clip_by_value(1/pred_,0.001,1)))
#    return loss
#
#def loss_func1(label,pred):
#    batch_array = (1 - label) + pred*label
#    batch_loss = tf.pow(tf.reduce_prod(batch_array,1),1/tf.reduce_sum(label,1))
#    loss = -tf.reduce_mean(tf.log(batch_loss))
#    return loss
#
#def compute_accuracy(v_xs, v_ys,output):
#    y_pre = sess.run(output, feed_dict={xs: v_xs})
#    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#    return result
#    
#def compute_correct_number(v_xs, v_ys,output):
#    y_pre = sess.run(output, feed_dict={xs: v_xs})
#    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#    correct_number = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
#    num = sess.run(correct_number, feed_dict={xs: v_xs, ys: v_ys})
#    return num
#
#def weight_variable(shape,init,stddev,name):
#    weight = tf.Variable(init(shape,stddev=stddev),name=name)
#    return weight
#
#def bias_variable(shape,name):
#    initial = tf.constant(0.1, shape=shape,name=name)
#    return tf.Variable(initial)
#
##定义权重偏置
#def define_WB(NetworkShape):
#    Weigth = []
#    Bias = []
#    for n in range(len(NetworkShape)-1):
#        n_in = NetworkShape[n]
#        n_out = NetworkShape[n+1]
#        W = weight_variable([n_in,n_out],tf.truncated_normal,0.1,'w'+str(n))
#        B = bias_variable([n_out],'b'+str(n))
#        Weigth.append(W)
#        Bias.append(B)
#    return Weigth,Bias
#
#def infer(xs,W,B):
#    layer1 = tf.nn.softmax(tf.matmul(xs,W[0]) + B[0])
#    return layer1
#
#def train(output,ys):
#    cross_entropy = loss_func2(ys,output) 
#    train_step = tf.train.AdamOptimizer(parse.lr).minimize(cross_entropy)
#    return train_step,cross_entropy
#
#
## define placeholder for inputs to network
#xs = tf.placeholder(tf.float32, [None, 7])   # 28x28
#ys = tf.placeholder(tf.float32, [None, 29])
#keep_prob = tf.placeholder(tf.float32)
#
#
#Weigth1,Bias1 = define_WB([7,29])
#Weigth2,Bias2 = define_WB([7,29])
#Weigth3,Bias3 = define_WB([7,29])
#Weigth4,Bias4 = define_WB([7,29])
#Weigth5,Bias5 = define_WB([7,29])
#
#output1 = infer(xs,Weigth1,Bias1)
#train_step1,cross_entropy1 = train(output1,ys)
#
#output2 = infer(xs,Weigth2,Bias2)
#train_step2,cross_entropy2 = train(output2,ys)
#
#output3 = infer(xs,Weigth3,Bias3)
#train_step3,cross_entropy3 = train(output3,ys)
#
#output4 = infer(xs,Weigth4,Bias4)
#train_step4,cross_entropy4 = train(output4,ys)
#
#output5 = infer(xs,Weigth5,Bias5)
#train_step5,cross_entropy5 = train(output5,ys)
#
#
#Abalone_data = AbaloneData.UCIData(parse.DataDir)
#X,Y,partial_Y = Abalone_data.createPartialData()
#
#item_data = CrossData(X)
#item_data.cross_split()
#
#correct_pred=[]
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
#Accu = sum(correct_pred)/item_data._num_examples
#print('5CrossAccu:',Accu)




