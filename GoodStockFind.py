# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:58:24 2018

@author: Administrator
"""

import tushare as ts
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import datetime

def WeightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def BiasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def Conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def MaxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#定义CNN
xs = tf.placeholder(tf.float32, [None, 80])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, 16, 5])

##conv2d layer =1#
W_conv1 = WeightVariable([1,2,5,10])
b_conv1 = BiasVariable([10])
h_conv1 = tf.nn.relu(Conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = MaxPool2x2(h_conv1)

##conv2d layer = 2#
W_conv2 = WeightVariable([1,2,10,20])
b_conv2 = BiasVariable([20])
h_conv2 = tf.nn.relu(Conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = MaxPool2x2(h_conv2)

#conv2d layer = 3#
W_conv3 = WeightVariable([1,2,20,40])
b_conv3 = BiasVariable([40])
h_conv3 = tf.nn.relu(Conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = MaxPool2x2(h_conv3)

#conv2d layer = 4#
W_conv4 = WeightVariable([1,2,40,80])
b_conv4 = BiasVariable([80])
h_conv4 = tf.nn.relu(Conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = MaxPool2x2(h_conv4)

## full connect layer =1#
W_fc1 = WeightVariable([1*1*80, 16])
b_fc1 = BiasVariable([16])
h_pool4_flat = tf.reshape(h_pool4, [-1, 1*1*80])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = WeightVariable([16, 2])
b_fc2 = BiasVariable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-7, 1.0)),
                                              reduction_indices=[1]))


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state('NetworkSaver/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Network Restore ok! ...')
    
stocklist_all = ts.get_today_all()
stocklist = list(stocklist_all.code)
namelist = list(stocklist_all.name)
stocktonames = {}
for i in range(0, len(stocklist)):
    code = stocklist[i]
    name = namelist[i]
    stocktonames[code] = name
print('\n')

def stockwithname(code):
    if code in stocktonames.keys():
        return '%s[%s]' %(code, stocktonames[code])
    return code

def isstockst(code):
    if code not in stocktonames.keys():
        return False
    name = stocktonames[code]
    if 'ST' in name:
        return True
    return False

def isrisestop(kdatapart):
    close1 = kdatapart['close'][len(kdatapart)-1]
    close2 = kdatapart['close'][len(kdatapart)-2]
    if close1/close2-1>0.098:
        return True
    return False

KDAYS = 16
TODAY_STR = str(date.today())
print('Today is %s' %TODAY_STR)

#如果是星期六或者星期天。则好回退到上周五的工作日
weekday = date.today().weekday()
if weekday>4:
    offsetday = weekday - 4
    TODAY_STR = str(date.today() - datetime.timedelta(days=offsetday))

sortStockList = []

count = 0
for stock in stocklist:
    try:
        df = ts.get_k_data(stock, index=False)
        l = len(df)
        if l<KDAYS:
            continue
        df = df[l-KDAYS:l]
        kdatapart = df.reset_index(drop=True)
        lastTradeDate = str(kdatapart['date'][len(kdatapart)-1])
        if TODAY_STR!=lastTradeDate:
            print('%s is paused Today! skip it!' %stockwithname(stock))
            continue
        if isstockst(stock) == True:
            print('%s is Special Treatment! skip it!' %stockwithname(stock))
            continue
        if isrisestop(kdatapart)==True:
            print('%s is Rise Stop! skip it!' %stockwithname(stock))
            continue
        lowlist = []
        highlist = []
        volumelist = []
        feeddata = []
        for j in range(0, len(kdatapart)):
            lowlist.append(float(kdatapart['low'][j]))
            highlist.append(float(kdatapart['high'][j]))
            volumelist.append(float(kdatapart['volume'][j]))
        low_min = min(lowlist)
        low_max = max(highlist)
        volume_min = min(volumelist)
        volume_max = max(volumelist)
        for j in range(0, len(kdatapart)):
            fopen = float(kdatapart['open'][j])
            fclose = float(kdatapart['close'][j])
            fhigh = float(kdatapart['high'][j])
            flow = float(kdatapart['low'][j])
            fvolume = float(kdatapart['volume'][j])
            unified_open = (fopen-low_min)/(low_max-low_min)
            unified_close = (fclose-low_min)/(low_max-low_min)
            unified_high = (fhigh-low_min)/(low_max-low_min)
            unified_low = (flow-low_min)/(low_max-low_min)
            unified_vol = (fvolume-volume_min)/(volume_max-volume_min)
            feeddata.append(unified_open)
            feeddata.append(unified_close)
            feeddata.append(unified_high)
            feeddata.append(unified_low)
            feeddata.append(unified_vol)
        inputData = np.array(feeddata).reshape(1, KDAYS*5)
        currPred = sess.run(prediction, feed_dict={xs:inputData, keep_prob:1})
        upPoss = currPred[0][0]
        downPoss = currPred[0][1]
        if count % 50 == 0:
            print('Statistics proceed %0.2f%%' %(float(count)/len(stocklist)*100.0))
        sortStockList.append([stock, upPoss])
    except:
        pass
    count+=1
    
#按照上涨概率排序（从小到大）

sortStockList.sort(key=lambda x:x[1], reverse=False)

low20 = sortStockList[0:20]
high20 = sortStockList[len(sortStockList)-20:len(sortStockList)]

print('Lowest Chance-----------------------------------')
for item in low20:
    print('%s Rise Chance Tomorrow : %0.2f%%' %(stockwithname(item[0]), item[1]*100.0))
    
print('Highest Chance-----------------------------------')
for item in high20:
    print('%s Rise Chance Tomorrow : %0.2f%%' %(stockwithname(item[0]), item[1]*100.0))    

    
    
    
    