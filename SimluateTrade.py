# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:02:07 2018

@author: Administrator
"""

import tushare as ts
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

#输入指数代码
indexCode = input('Input Code:')
#股票或者指数
type = input('1-Stock 2-Index:')
if type=='1':
    isIndex = False
else:
    isIndex = True

#输入开始日期
startDate = input('Input Start Date:')
#输入结束日期
endDate = input('Input End Date:')

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
    
df = ts.get_k_data(indexCode, index=isIndex, start=startDate, end=endDate)
del df['date']
del df['code']
totalline = len(df)
if totalline<65:
    print('Too few data, cannot sim trading!')
    exit()
    
KDAYS = 16

groundTruthList = []
feeddatalist = []

for i in range(0, totalline-KDAYS):
    groundTruthList.append(float(df['close'][i+KDAYS]) / float(df['close'][i+(KDAYS-1)]))    
    kdatapart = df[i:i+KDAYS]
    kdatapart = kdatapart.reset_index(drop=True)
    lowlist = []
    volumelist = []
    feeddata = []
    for j in range(0, len(kdatapart)):
        lowlist.append(float(kdatapart['low'][j]))
        volumelist.append(float(kdatapart['volume'][j]))
    low_min = min(lowlist)
    low_max = max(lowlist)
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
    feeddatalist.append(feeddata)

benchmark_netvalue = 1.0
simtrade_netvalue = 1.0
simtrade_poslevel = 0.0
benchmark_netvalue_list = []
simtrade_netvalue_list = []
upPoss_list = []
alpha_list = []        
has_position = False
predictRight = 0.0
predictTotal = 0.000001

BUY_LINE = 0.6
TRADE_COST = 0.00025
TAX_COST = 0.001

for i in range(0, len(groundTruthList)):
    growth = groundTruthList[i]
    feeddata = feeddatalist[i]
    inputData = np.array(feeddata).reshape(1, KDAYS*5)
    currPred = sess.run(prediction, feed_dict={xs:inputData, keep_prob:1})
    choice = sess.run(tf.argmax(currPred,1))[0]
    upPoss = currPred[0][0]
    upPoss_list.append(upPoss)
    newGrowth = growth-1.0
    if upPoss>=BUY_LINE and newGrowth>0.0:
        predictRight += 1.0
    
    benchmark_netvalue = benchmark_netvalue * growth
    benchmark_netvalue_list.append(benchmark_netvalue)
    if has_position==False:
        if upPoss>BUY_LINE:
            has_position=True
            simtrade_netvalue -= simtrade_netvalue * TRADE_COST
    else:
        if upPoss<BUY_LINE:
            if has_position==True:
                simtrade_netvalue -= simtrade_netvalue * TRADE_COST
                simtrade_netvalue -= simtrade_netvalue * TAX_COST
            has_position=False
    if has_position==True:
        simtrade_netvalue = simtrade_netvalue * growth
    simtrade_netvalue_list.append(simtrade_netvalue)
    alpha_list.append(simtrade_netvalue/benchmark_netvalue-1)
    
    if upPoss>=BUY_LINE:
        predictTotal += 1.0
    
    if i%30==0:
        percent = i/float(len(groundTruthList))
        print('Simulate Calc %0.2f%% ...' %(percent*100.0))
    
plt.figure(figsize=(12,7))
plt.title('%s Sim Trade Net Value Chart\n' %indexCode)
plt.xlabel('Days')
plt.ylabel('NetValue')
plt.plot(simtrade_netvalue_list, linewidth=1.0, color=[1,0,0], label='SimTrade')
plt.plot(benchmark_netvalue_list, linewidth=1.0, color=[0,0,1], label='Benchmark')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(12,7))
plt.title('%s Alpha Chart\n' %indexCode)
plt.xlabel('Days')
plt.ylabel('Alpha')
plt.plot(alpha_list, linewidth=1.0, color=[1,0,0], label='Alpha')
plt.legend(loc='upper left')
plt.show()

print('Accuracy : %0.2f%%' %(predictRight/predictTotal*100.0))
print('Benchmark NetValue : %f' %(benchmark_netvalue))
print('Simtrade NetValue : %f' %(simtrade_netvalue))