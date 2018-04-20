# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:26:32 2018

@author: yewei
"""

import tushare as ts
from datetime import date
import pandas as pd

defaultStartDate = '2007-01-01'
defaultEndDate = str(date.today())

code = input("Input Index Code : ")
startDate = input("Input Start Date (%s) : " %defaultStartDate)
endDate = input("Input End Date (%s) : " %defaultEndDate)

if startDate=="":
    startDate = defaultStartDate
if endDate=="":
    endDate = defaultEndDate
    
outputFileName = 'DataDownload/Data_%s_(%s)_(%s).csv' %(code, startDate, endDate)
print('Data Download to : %s' %outputFileName)

df = ts.get_k_data(code, index=True, start=str(startDate), end=str(endDate))
del df['date']
del df['code']

rows = len(df)

print('Data has %d Rows ... ' %rows)

#data process
TRAIN_DAYS = 64
RANGE_END = rows-TRAIN_DAYS

if rows<=TRAIN_DAYS:
    exit()
    
table = pd.DataFrame()
for i in range(0, TRAIN_DAYS):
    colname1 = 'O%d' %i
    colname2 = 'C%d' %i
    colname3 = 'H%d' %i
    colname4 = 'L%d' %i
    colname5 = 'V%d' %i
    table[colname1] = None
    table[colname2] = None
    table[colname3] = None
    table[colname4] = None
    table[colname5] = None
table['UP'] = None
table['DOWN'] = None

for i in range(0, RANGE_END):
    growth = float(df['close'][i+64]) / float(df['close'][i+63]) - 1
    kdatapart = df[i:i+64]
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
    up = 1.0
    down = 0.0
    if growth*100.0>=0.0:
        up = 1.0
        down = 0.0
    else:
        up = 0.0
        down = 1.0
    feeddata.append(up)
    feeddata.append(down)
    
    table.loc[len(table.index)] = feeddata
    
    if i%100==0:
        percent = i / float(RANGE_END) * 100.0
        print("Exporting %0.2f%% ..." %percent)

print('Saving To %s ...' %outputFileName)

table = table.dropna(axis=0, how='any')
table = table.drop_duplicates()

table.to_csv(path_or_buf=outputFileName, index=False, header=True)

print('Data Saved ...')