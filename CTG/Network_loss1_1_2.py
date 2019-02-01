# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:11:14 2019

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

CTGData = imp.load_source('CTG','../Data.py')



parser = argparse.ArgumentParser(description="show some parse for the progess")
parser.add_argument("--DataDir",type=str,default="../uci_data_set/CTG.mat",help="the mat data file path")
parser.add_argument("--train_epoch",type=int,default=5000,help="number of training epochs")
parser.add_argument("--model_save_name",type=str,default="Abalone.ckpt",help="the file name for model to save")
parser.add_argument("--model_save_dir",type=str,default=None,help="the dir for model to save")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
parser.add_argument("--save2txt", default=1, type=int, choices=[0,1], help="Whether to save result to txt")
#parser.add_argument("--use_gpu_nb", default=1, type=int, choices=[0,1], help="the number of GPU to use")
parser.add_argument("-npt","--nb_partial_target", type=int, default=1, choices=[1,2,3], help="number of partial target without count the true target")
parser.add_argument("-cpdr","--create_partial_data_rate", type=float,default=0.5, help="how much date transpose to partial data ,the value is (0,1)")
parse = parser.parse_args()

#if parse.use_gpu:
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(parse.use_gpu_nb)
#else:
#    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if parse.create_partial_data_rate < 0 or parse.create_partial_data_rate > 1:
    raise ValueError("parameter create_partial_data_rate must be range from 0 to 1")
    
createVar = globals()


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


def getGraph(cv=5):
    output_array = []
    cross_entropy_array = []
    train_step_array = []
    Graph_array = []
    xs_array = []
    ys_array = []

    for i in range(cv):
        createVar["g"+str(i)] = tf.Graph()
        with createVar["g"+str(i)].as_default():
            with tf.variable_scope("Graph{}".format(i)):
                createVar["xs_g"+str(i)] = tf.placeholder(tf.float32, [None, 21],name='data')
                createVar["ys_g"+str(i)] = tf.placeholder(tf.float32, [None, 10],name='target')
                createVar["weight_g"+str(i)] = weight_variable([21,10],tf.truncated_normal,0.1,'w0')
                createVar["bais_g"+str(i)] = bias_variable([10],'b0')
                createVar["output_g"+str(i)] = tf.matmul(createVar["xs_g"+str(i)],createVar["weight_g"+str(i)]) + createVar["bais_g"+str(i)]
                createVar["cross_entropy_g"+str(i)] = loss_func(createVar["ys_g"+str(i)],createVar["output_g"+str(i)])
                createVar["train_step_g"+str(i)] = tf.train.AdamOptimizer(parse.lr).minimize(createVar["cross_entropy_g"+str(i)])
                output_array.append(createVar["output_g"+str(i)])
                cross_entropy_array.append(createVar["cross_entropy_g"+str(i)])
                train_step_array.append(createVar["train_step_g"+str(i)])
                xs_array.append(createVar["xs_g"+str(i)])
                ys_array.append(createVar["ys_g"+str(i)])
        Graph_array.append(createVar["g"+str(i)])
    return output_array,cross_entropy_array,train_step_array,Graph_array,xs_array,ys_array

CTG_data = CTGData.UCIData(parse.DataDir)
X,Y,partial_Y = CTG_data.createPartialData(rate=parse.create_partial_data_rate,nb_partial_target=parse.nb_partial_target)
X = Data.guiYiHua(X)


Accu_array = []

for N in range(10):


    item_data = CrossData(X)
    item_data.cross_split()
    
    output_array,cross_entropy_array,train_step_array,Graph_array,xs_array,ys_array = getGraph()
    
    correct_pred = []
    
    for j in range(5):
    #    createVar["sess"+str(j)] = tf.Session(graph=Graph_array[j])
        with tf.Session(graph=Graph_array[j]) as sess:
            sess.run(tf.global_variables_initializer())
            for k in range(parse.train_epoch):
                batch_index= item_data.next_batch(parse.batch_size,j)
                batch_xs, batch_ys, batch_ys_ = X[batch_index],Y[batch_index],partial_Y[batch_index]
                batch_ys_ = Data.deal_data(batch_ys_)
                sess.run(train_step_array[j], feed_dict={xs_array[j]: batch_xs, ys_array[j]: batch_ys_})
                if k % 50 == 0:
                    print("training {}/{}:".format(N,j),k/50)
                    print("train:",compute_accuracy(xs_array[j],ys_array[j],batch_xs,batch_ys,output_array[j],sess))
                    print("test:",compute_accuracy(xs_array[j],ys_array[j],
                        X[item_data.data_test[0]], Y[item_data.data_test[0]], output_array[j],sess))
                    print("loss:",sess.run(cross_entropy_array[j],feed_dict={xs_array[j]:batch_xs, ys_array[j]:batch_ys_}))
            corr_num = compute_correct_number(xs_array[j],ys_array[j],X[item_data.data_test[0]], Y[item_data.data_test[0]], output_array[j],sess)
            correct_pred.append(corr_num)
            item_data.set_index_to_zero()
    Accu = sum(correct_pred)/item_data._num_examples
    Accu_array.append(Accu)
print(Accu_array)
if parse.save2txt:
    CTGData.UCIData.save2txt(parse,Accu_array,'Network_loss1_1_2.txt')





