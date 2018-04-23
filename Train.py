# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:25:23 2018

@author: yewei
"""
import tensorflow as tf
import matplotlib.pyplot as plt

from DataSet import TrainDataSet
from DataSet import TestDataSet

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


def ComputeAccuracy(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(v_ys * tf.log(tf.clip_by_value(y_pre, 1e-7, 1.0)),
                                              reduction_indices=[1]))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    result1 = sess.run(cross_entropy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result,result1

#获得验证集
myTestData = TestDataSet('TestingData', 320, 2)
testData1,labelData1 = myTestData.GetData()

#获得训练集
myTrainData = TrainDataSet('TrainingData', 320, 2)

#开始训练按ENTER继续
input("\nPress ENTER key to continue...")

#定义CNN
xs = tf.placeholder(tf.float32, [None, 320])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, 64, 5])

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

#conv2d layer = 5#
#W_conv5 = WeightVariable([1,2,80,160])
#b_conv5 = BiasVariable([160])
#h_conv5 = tf.nn.relu(Conv2d(h_pool4, W_conv5) + b_conv5)
#h_pool5 = MaxPool2x2(h_conv5)

## full connect layer =1#
W_fc1 = WeightVariable([1*4*80, 128])
b_fc1 = BiasVariable([128])
h_pool4_flat = tf.reshape(h_pool4, [-1, 1*4*80])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = WeightVariable([128, 2])
b_fc2 = BiasVariable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-7, 1.0)),
                                              reduction_indices=[1]))

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-4, global_step, 10000, 0.97)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

saver = tf.train.Saver()

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state('NetworkSaver/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Network Restore ok! ...')

train_cross_entropy_list = []
test_cross_entropy_list = []

def TrainingDataProcess(trainData, labelData, rowReaded, epochCount, trainCount):
    sess.run(train_step, feed_dict={xs: trainData, ys: labelData, keep_prob:0.0625})
    if trainCount % 50 == 0:
        trainSetAccuracy,train_cross_entropy = ComputeAccuracy(trainData,labelData)
        testSetAccuracy,test_cross_entropy = ComputeAccuracy(testData1,labelData1)
        trainSetAccuracy*=100
        testSetAccuracy*=100
        train_cross_entropy_list.append(train_cross_entropy)
        test_cross_entropy_list.append(test_cross_entropy)
        if len(train_cross_entropy_list)>1000:
            del train_cross_entropy_list[0]
        if len(test_cross_entropy_list)>1000:
            del test_cross_entropy_list[0]
        print('BatchCount=%d , Epoch=%d , Accuracy(TrainSet:%0.2f%% [%f] , TestSet:%0.2f%% [%f])' %(trainCount, epochCount, trainSetAccuracy, train_cross_entropy, testSetAccuracy, test_cross_entropy))
    if trainCount % 3001 == 0:
        #保持网络
        SaverNetwork()
        Plot()
        
def SaverNetwork():
    print('Saving ...')
    save_path = saver.save(sess, 'NetworkSaver/model.ckpt', write_meta_graph=False)
    print("Saved to : ", save_path)
    lr = sess.run(learning_rate)
    print('Curr Learning Rate : %f' %lr)

def Plot():
    plt.figure(figsize=(12,7))
    plt.title('Cross Entropy Record')
    plt.xlabel('Step')
    plt.ylabel('Cross Entropy')
    plt.plot(train_cross_entropy_list, color=[1,0,0], label='TrainSet')
    plt.plot(test_cross_entropy_list, color=[0,0,1], label='TestSet')
    plt.legend()
    plt.show()
    plt.close()


myTrainData.LoopData(100, 1000, TrainingDataProcess)


exit