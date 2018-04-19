# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:11:53 2018

@author: yewei
"""

import numpy as np
import pandas as pd
import glob
import random

def DefaultDataProcess(trainData, labelData, rowReaded, epochCount, trainCount):
    pass

class TrainDataSet(object):
    def __init__(self, dataPath, dataSizeCount, labelSizeCount):
        self.dataPath = dataPath
        self.dataSizeCount = dataSizeCount
        self.labelSizeCount = labelSizeCount
        self.fileList = []
        for filename in glob.glob(dataPath+"\\*.csv"):
            self.fileList.append(filename)
            print('TrainDataSet file : %s' %filename)
        random.shuffle(self.fileList)   
            
    def LoopData(self, rowCount, maxEpoch, dataProcessFunc=DefaultDataProcess):
        trainCount = 0
        for epoch in range(0, maxEpoch):
            for filename in self.fileList:
                df = pd.DataFrame()
                df = pd.read_csv(filename)
                df.dropna(axis=0, how='any')
                df = df.drop_duplicates()
                totalline = len(df)
                indexlist = list(range(0, totalline))
                random.shuffle(indexlist)
                df.index = indexlist
                df = df.sort_index() 
               
                for i in range(0, totalline, rowCount):
                    step=rowCount
                    if i+rowCount<totalline:
                        step=rowCount
                    else:
                        step=totalline-i
                    line = df[i:i+step]
                    trainData = np.array(line.iloc[0:step,
                                                   0:self.dataSizeCount]).reshape(step,self.dataSizeCount)
                    labelData = np.array(line.iloc[0:step,
                                                   self.dataSizeCount:self.dataSizeCount+self.labelSizeCount]).reshape(step,self.labelSizeCount)
    
                    dataProcessFunc(trainData, labelData, step, epoch+1, trainCount)
                    trainCount+=1


class TestDataSet(object):
    def __init__(self, dataPath, dataSizeCount, labelSizeCount):
        self.dataPath = dataPath
        self.dataSizeCount = dataSizeCount
        self.labelSizeCount = labelSizeCount
        self.fileList = []
        self.totaldf = pd.DataFrame()
        for filename in glob.glob(dataPath+"\\*.csv"):
            self.fileList.append(filename)
            print('TestDataSet : %s' %filename)
            df = pd.DataFrame()
            df = pd.read_csv(filename)
            df.dropna(axis=0, how='any')
            df = df.drop_duplicates()
            self.totaldf = self.totaldf.append(df)
            
    def GetData(self):
        totalline = len(self.totaldf)
        testData = np.array(self.totaldf.iloc[0:totalline,
                                              0:self.dataSizeCount]).reshape(totalline,self.dataSizeCount)
        labelData = np.array(self.totaldf.iloc[0:totalline,
                                               self.dataSizeCount:self.dataSizeCount+self.labelSizeCount]).reshape(totalline,self.labelSizeCount)
        return testData,labelData
    
