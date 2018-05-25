# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:02:07 2018

@author: Administrator
"""

import tushare as ts
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

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
type = input('1-Stock(Default) 2-Index:')
if type=='2':
    isIndex = True
else:
    isIndex = False

defaultStartDate = '1997-01-01'
defaultEndDate = '2005-01-01'

#输入开始日期
startDate = input('Input Start Date (%s as Default):' %defaultStartDate)
#输入结束日期
endDate = input('Input End Date (%s as Default):' %defaultEndDate)

if startDate=='':
    startDate=defaultStartDate
if endDate=='':
    endDate=defaultEndDate

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

for i in range(0, totalline-KDAYS-1):
    groundTruthList.append(float(df['close'][i+KDAYS]) / float(df['close'][i+KDAYS-1]))    
    kdatapart = df[i:i+KDAYS]
    kdatapart = kdatapart.reset_index(drop=True)
    lowlist = []
    highlist = []
    volumelist = []
    feeddata = []
    
    lowpart = kdatapart['low']
    highpart = kdatapart['high']
    volpart = kdatapart['volume']
    openpart = kdatapart['open']
    closepart = kdatapart['close']
    
    for j in range(0, len(kdatapart)):
        lowlist.append(float(lowpart[j]))
        highlist.append(float(highpart[j]))
        volumelist.append(float(volpart[j]))
    low_min = min(lowlist)
    low_max = max(highlist)
    volume_min = min(volumelist)
    volume_max = max(volumelist)
    for j in range(0, len(kdatapart)):
        fopen = float(openpart[j])
        fclose = float(closepart[j])
        fhigh = float(highpart[j])
        flow = float(lowpart[j])
        fvolume = float(volpart[j])
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

BUY_LINE = 0.75

BUY_LINE_CLUSTER        = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, -1] 
HAS_POSITION_CLUSTER    = [False, False, False, False, False, False, False, False, False, False]
NET_VALUE_CLUSTER       = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
NET_VALUE_LIST_CLUSTER  = [[], [], [], [], [], [], [], [], [], []]
LOSE_WEIGHT_CLUSTER     = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
WIN_WEIGHT_CLUSTER      = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
TRADE_DAYS_CLUSTER      = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
CLUSTER_COUNT           = len(BUY_LINE_CLUSTER)

TRADE_COST = 0.00025
TAX_COST = 0.001

LOSE_WEIGHT = 0.00001
WIN_WEIGHT = 0.0
HIT_COUNT = 0.0
TOTAL_COUNT = 0.00001

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
        
    if upPoss>=BUY_LINE and newGrowth>0.0:
        WIN_WEIGHT+=newGrowth*simtrade_netvalue
    if upPoss>=BUY_LINE and newGrowth<0.0:
        LOSE_WEIGHT+=(-newGrowth)*simtrade_netvalue
        
    TOTAL_COUNT+=1.0
    if upPoss>=BUY_LINE:
        HIT_COUNT+=1.0
    
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
        
    #calc cluster
    for j in range(0, CLUSTER_COUNT):
        buyLine = BUY_LINE_CLUSTER[j]
        if buyLine == -1:
            buyLine = random.random()
            upPoss = random.random()
        
        if upPoss>=buyLine and newGrowth>0.0:
            WIN_WEIGHT_CLUSTER[j]+=newGrowth*NET_VALUE_CLUSTER[j]
        if upPoss>=buyLine and newGrowth<0.0:
            LOSE_WEIGHT_CLUSTER[j]+=(-newGrowth)*NET_VALUE_CLUSTER[j]
            
        if upPoss>=buyLine:
            TRADE_DAYS_CLUSTER[j]+=1.0
            
        if HAS_POSITION_CLUSTER[j]==False:
            if upPoss>buyLine:
                HAS_POSITION_CLUSTER[j]=True
                NET_VALUE_CLUSTER[j] -= NET_VALUE_CLUSTER[j] * TRADE_COST
        else:
            if upPoss<buyLine:
                if HAS_POSITION_CLUSTER[j]==True:
                    NET_VALUE_CLUSTER[j] -= NET_VALUE_CLUSTER[j] * TRADE_COST
                    NET_VALUE_CLUSTER[j] -= NET_VALUE_CLUSTER[j] * TAX_COST
                HAS_POSITION_CLUSTER[j]=False
        if HAS_POSITION_CLUSTER[j]==True:
            NET_VALUE_CLUSTER[j] = NET_VALUE_CLUSTER[j] * growth
        NET_VALUE_LIST_CLUSTER[j].append(NET_VALUE_CLUSTER[j])
        
    simtrade_netvalue_list.append(simtrade_netvalue)
    alpha_list.append(simtrade_netvalue/benchmark_netvalue-1)
    
    if upPoss>=BUY_LINE:
        predictTotal += 1.0
    
    if i%30==0:
        percent = i/float(len(groundTruthList))
        print('Simulate Calc %0.2f%% ...' %(percent*100.0))
    
plt.figure(figsize=(15,10))
plt.title('%s Sim Trade Net Value Chart\n' %indexCode)
plt.xlabel('Days')
plt.ylabel('NetValue')
plt.plot(simtrade_netvalue_list, linewidth=5.0, color=[1,0,0], label='SimTrade(%0.2f)' %BUY_LINE)
plt.plot(benchmark_netvalue_list, linewidth=1.0, color=[0,0,1], linestyle='--', label='Benchmark')

for j in range(0, CLUSTER_COUNT):
    lw = 1.0
    rankstr = ''
    if NET_VALUE_CLUSTER[j]==max(NET_VALUE_CLUSTER):
        lw = 2.0
        rankstr = '(Best)'
    elif NET_VALUE_CLUSTER[j]==min(NET_VALUE_CLUSTER):
        lw = 2.0
        rankstr = '(Worst)'
    if BUY_LINE_CLUSTER[j]!=-1:
        plt.plot(NET_VALUE_LIST_CLUSTER[j], linewidth=lw, label='bl %0.2f %s' %(BUY_LINE_CLUSTER[j], rankstr))
    else:
        plt.plot(NET_VALUE_LIST_CLUSTER[j], linewidth=3.0, linestyle=':', label='RandomSet')
plt.legend(loc='upper left')
plt.show()

#plt.figure(figsize=(12,7))
#plt.title('%s Alpha Chart\n' %indexCode)
#plt.xlabel('Days')
#plt.ylabel('Alpha')
#plt.plot(alpha_list, linewidth=1.0, color=[1,0,0], label='Alpha')
#plt.legend(loc='upper left')
#plt.show()

print('Accuracy : %0.2f%%' %(predictRight/predictTotal*100.0))
print('Profit&loss Ratio : %0.2f%%' %((WIN_WEIGHT/LOSE_WEIGHT-1)*100))
print('Hit Rate : %0.2f%%' %(HIT_COUNT/TOTAL_COUNT*100.0))
print('Benchmark NetValue : %f' %(benchmark_netvalue))
print('Simtrade NetValue : %f' %(simtrade_netvalue))
print('\n')
for j in range(0, CLUSTER_COUNT):
    print('Profit&loss Ratio For Cluster(%0.2f) : %0.2f%%' %(BUY_LINE_CLUSTER[j], (WIN_WEIGHT_CLUSTER[j]/LOSE_WEIGHT_CLUSTER[j]-1)*100))
    
print('\n')
for j in range(0, CLUSTER_COUNT):
    print('Profit Rate PerDay (%d/%d) For Cluster(%0.2f) : %0.2f%%' %(TRADE_DAYS_CLUSTER[j],int(TOTAL_COUNT), BUY_LINE_CLUSTER[j], (NET_VALUE_CLUSTER[j]-1)/TRADE_DAYS_CLUSTER[j]*100.0))
    
