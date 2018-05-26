# -*- coding: utf-8 -*-
"""
Created on Sat May 26 17:34:33 2018

@author: Administrator
"""
import tushare as ts
import tensorflow as tf
import numpy as np
import pandas as pd

TEST_CODES = [u'000625', u'000629', u'600783', u'600662', u'000021', u'000022', u'600788', u'000024', u'000027', u'600665', u'600988', u'600985', u'600984', u'600987', u'600986', u'600981', u'600980', u'600983', u'600982', u'600426', u'600425', u'600360', u'600422', u'600351', u'600893', u'600420', u'600892', u'600421', u'600891', u'600722', u'600897', u'000866', u'600896', u'600895', u'600358', u'600894', u'600359', u'600293', u'600429', u'600857', u'600856', u'600855', u'600854', u'600853', u'600852', u'600851', u'600850', u'600859', u'600858', u'600581', u'600580', u'600583', u'600582', u'600585', u'600584', u'600587', u'600586', u'600589', u'600588', u'600611', u'600610', u'600613', u'600612', u'600615', u'600614', u'600617', u'600616', u'600619', u'600618', u'600758', u'600759', u'600750', u'600751', u'600752', u'600753', u'600754', u'600755', u'600756', u'600757', u'600039', u'600038', u'600031', u'600030', u'600033', u'600035', u'600037', u'600036', u'600535', u'600536', u'600537', u'600530', u'600531', u'600532', u'600533', u'600538', u'600539', u'600109', u'600108', u'600105', u'600104', u'600107', u'600106', u'600101', u'600100', u'600103', u'600102', u'600325', u'600327', u'600249', u'600321', u'600320', u'600323', u'600322', u'600242', u'600243', u'000817', u'600241', u'600329', u'600247', u'600400', u'600401', u'600403', u'600405', u'600406', u'600408', u'600409', u'000709', u'600216', u'600299', u'600060', u'600099', u'600887', u'600098', u'600502', u'600091', u'600159', u'000039', u'600298', u'600790', u'600658', u'600798', u'600799', u'600378', u'600076', u'600328', u'600239', u'600884', u'600371', u'600885', u'600079', u'600639', u'600377', u'600822', u'600823', u'600820', u'600821', u'600826', u'600827', u'600824', u'600825', u'600880', u'600828', u'600829', u'600737', u'600882', u'600883', u'600738', u'600739', u'600637', u'600889', u'600747', u'600746', u'600745', u'600744', u'600628', u'600629', u'600741', u'600740', u'600624', u'600626', u'600627', u'600620', u'600621', u'600622', u'600748', u'600523', u'600522', u'600521', u'600520', u'600527', u'600526', u'600525', u'600480', u'600481', u'600482', u'600483', u'600485', u'600486', u'600487', u'600138', u'600139', u'600130', u'600131', u'600132', u'600133', u'600135', u'600136', u'600137', u'600310', u'600311', u'600312', u'600313', u'600315', u'600316', u'600317', u'600251', u'600250', u'600253', u'600252', u'600255', u'000825', u'600257', u'600256', u'000550', u'600570', u'600571', u'600004', u'600557', u'600006', u'600007', u'600000', u'600001', u'600002', u'600003', u'600416', u'600415', u'600008', u'600009', u'600558', u'600410', u'600579', u'600123', u'600089', u'600460', u'600129', u'600128', u'600467', u'600505', u'600831', u'600830', u'600833', u'600832', u'600835', u'600834', u'600837', u'600836', u'600839', u'600838', u'600063', u'000541', u'600308', u'600303', u'600068', u'600743', u'600300', u'600742', u'600732', u'600733', u'600886', u'600638', u'600736', u'600881', u'600734', u'600735', u'600633', u'600631', u'600630', u'600888', u'600636', u'600635', u'600634', u'600497', u'600496', u'600495', u'600493', u'600749', u'600491', u'600490', u'600623', u'600499', u'600498', u'600127', u'600126', u'600125', u'600088', u'600122', u'600121', u'600120', u'600084', u'600085', u'600086', u'600087', u'600080', u'600081', u'600082', u'600083', u'600648', u'600649', u'600647', u'600644', u'600645', u'600642', u'600643', u'600640', u'600641', u'600066', u'600067', u'600064', u'600065', u'600062', u'000429', u'600309', u'600061', u'600307', u'600306', u'600305', u'000932', u'600302', u'600301', u'600069', u'600545', u'600012', u'600547', u'600546', u'600016', u'600543', u'600390', u'600391', u'600392', u'600461', u'600466', u'600548', u'600396', u'600397', u'000725', u'600398', u'600399', u'600468', u'600050', u'600265', u'600266', u'600267', u'600260', u'600261', u'600262', u'600263', u'600268', u'600269', u'000839', u'600540', u'600015', u'600462', u'600463', u'600019', u'600393', u'600549', u'600395', u'600419', u'600418', u'600488', u'600489', u'600529', u'600528', u'600010', u'600808', u'600809', u'600804', u'600805', u'600806', u'600807', u'600800', u'600801', u'600802', u'600803', u'600259', u'600258', u'600721', u'600720', u'600723', u'600890', u'600725', u'600724', u'600727', u'600726', u'600729', u'600728', u'600899', u'600898', u'600318', u'600319', u'600152', u'600153', u'600150', u'600151', u'600156', u'600157', u'600155', u'600093', u'600092', u'600158', u'600090', u'600097', u'600096', u'600095', u'600094', u'600794', u'600795', u'600796', u'600797', u'600659', u'600791', u'600792', u'600793', u'600655', u'600654', u'600657', u'600656', u'600651', u'600650', u'600653', u'600652', u'600075', u'600074', u'600077', u'600379', u'600071', u'600070', u'600073', u'600072', u'600372', u'600373', u'600370', u'000539', u'600376', u'600078', u'600375', u'600011', u'600387', u'600386', u'600385', u'600383', u'600382', u'600381', u'600380', u'600389', u'600388', u'600479', u'600478', u'600572', u'600573', u'600575', u'600576', u'600577', u'600578', u'600470', u'600475', u'600477', u'600476', u'600206', u'600207', u'600205', u'600202', u'600203', u'600200', u'600201', u'600208', u'600209', u'600111', u'600273', u'600272', u'600271', u'600270', u'600277', u'600276', u'600275', u'600279', u'600278', u'600556', u'600005', u'600555', u'600552', u'000959', u'600550', u'600551', u'000066', u'000063', u'000068', u'000069', u'000100', u'600559', u'600975', u'600976', u'600971', u'600973', u'600978', u'600979', u'600819', u'600818', u'600812', u'600811', u'600810', u'600817', u'600816', u'600815', u'600814', u'600718', u'600719', u'600868', u'600869', u'600866', u'600715', u'600864', u'600865', u'600710', u'600863', u'600860', u'600713', u'600248', u'600326', u'000406', u'600141', u'600143', u'600145', u'600146', u'600149', u'600148', u'600668', u'600782', u'600781', u'600780', u'600787', u'600786', u'600785', u'600784', u'600660', u'600661', u'600789', u'600663', u'600664', u'600240', u'600666', u'600667', u'600369', u'600368', u'600361', u'600246', u'600363', u'600362', u'600365', u'600169', u'600367', u'600366', u'600601', u'600444', u'600446', u'600448', u'600449', u'600569', u'600568', u'600567', u'600566', u'600565', u'600563', u'600562', u'600561', u'600560', u'600215', u'600217', u'600609', u'600211', u'600210', u'600213', u'600212', u'600219', u'600218', u'000983', u'600288', u'600289', u'600286', u'600287', u'600284', u'600285', u'600282', u'600283', u'600280', u'600281', u'000778', u'600187', u'600029', u'600469', u'600026', u'600331', u'000002', u'000001', u'000009', u'600703', u'600874', u'600877', u'600700', u'600871', u'600706', u'600963', u'600962', u'600961', u'600960', u'600967', u'600966', u'600965', u'600969', u'600872', u'600873', u'000956', u'600708', u'600879', u'600338', u'600875', u'600702', u'600701', u'600876', u'600707', u'600870', u'600705', u'600704', u'600688', u'600689', u'600682', u'600683', u'600680', u'600681', u'600686', u'600687', u'600684', u'600685', u'600501', u'600500', u'600503', u'600677', u'600676', u'600675', u'600674', u'600178', u'600179', u'600671', u'600175', u'600176', u'600177', u'600170', u'600171', u'600172', u'600678', u'600507', u'600778', u'600779', u'600776', u'600777', u'600774', u'600775', u'600772', u'600773', u'600770', u'600771', u'600196', u'600197', u'600195', u'600192', u'600193', u'600190', u'600191', u'600057', u'600056', u'600055', u'600054', u'600053', u'600052', u'600198', u'600199', u'600452', u'600456', u'600455', u'600459', u'600458', u'600518', u'600519', u'600512', u'600513', u'600510', u'600511', u'600516', u'600517', u'600515', u'600220', u'600221', u'600222', u'600223', u'600225', u'600226', u'600227', u'600228', u'600229', u'000581', u'600231', u'600673', u'600339', u'600672', u'600354', u'600355', u'600356', u'600357', u'600350', u'600423', u'600352', u'600353', u'600295', u'600297', u'600296', u'600291', u'600290', u'600428', u'600292', u'000858', u'600679', u'600173', u'000088', u'000089', u'600608', u'600343', u'600059', u'000012', u'600058', u'600730', u'600436', u'600345', u'600731', u'600051', u'600997', u'600995', u'600992', u'600993', u'600990', u'600991', u'600714', u'600867', u'600716', u'600717', u'600862', u'600711', u'600712', u'600861', u'600900', u'600848', u'600840', u'600841', u'600842', u'600843', u'600844', u'600845', u'600846', u'600847', u'600699', u'600698', u'600691', u'600690', u'600693', u'600692', u'600695', u'600694', u'600697', u'600696', u'600592', u'600593', u'600590', u'600591', u'600596', u'600597', u'600594', u'600595', u'600598', u'600599', u'600602', u'600603', u'600600', u'600168', u'600606', u'600607', u'600604', u'600605', u'600163', u'600162', u'600161', u'600160', u'600167', u'600166', u'600165', u'000898', u'600769', u'600768', u'600765', u'600764', u'600767', u'600766', u'600761', u'600760', u'600763', u'600762', u'600185', u'600184', u'600028', u'600186', u'600181', u'600180', u'600183', u'600182', u'600022', u'600020', u'600021', u'600189', u'600188', u'600553', u'600509', u'600508', u'600118', u'600119', u'600116', u'600117', u'600114', u'600115', u'600112', u'600113', u'600110', u'600506', u'600336', u'600337', u'600335', u'600332', u'600333', u'600330', u'600238', u'600237', u'600236', u'600235', u'600234', u'600233', u'600232', u'000800', u'600230', u'600435', u'600340', u'600346', u'600433', u'600432', u'600348', u'600439', u'600438', u'000717', u'000488']

DEFAULT_START_DATE = '1997-01-01'
DEFAULT_END_DATE = '2005-01-01'
KDAYS = 16
BUY_LINE = 0.70
TRADE_COST = 0.00025
TAX_COST = 0.001
OUTPUT_FILENAME = 'TestReport/Report.csv'

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
W_fc1 = WeightVariable([1*1*80, 32])
b_fc1 = BiasVariable([32])
h_pool4_flat = tf.reshape(h_pool4, [-1, 1*1*80])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = WeightVariable([32, 2])
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

#生成表头
table = pd.DataFrame()
table['股票代码'] = None
table['准确率'] = None
table['盈亏比'] = None
table['命中率'] = None
table['基准净值'] = None
table['交易净值'] = None
table['每日平均收益'] = None

table.to_csv(path_or_buf=OUTPUT_FILENAME, index=False, header=True)

for code in TEST_CODES:
    df = ts.get_k_data(code, index=False, start=DEFAULT_START_DATE, end=DEFAULT_END_DATE)
    del df['date']
    del df['code']
    totalline = len(df)
    if totalline<500:
        print('%s Too few data, cannot sim trading!' %code)
        continue
    
    print('Calc For %s' %code)
    
    groundTruthList = []
    feeddatalist = []
    
    for i in range(0, totalline-KDAYS-1):
        #groundTruthList.append(float(df['close'][i+KDAYS]) / float(df['close'][i+KDAYS-1]))
        groundTruthList.append(float(df['open'][i+KDAYS+1]) / float(df['open'][i+KDAYS]))    
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
   
    LOSE_WEIGHT = 0.00001
    WIN_WEIGHT = 0.0
    HIT_COUNT = 0.0
    TOTAL_COUNT = 0.00001
    TRADE_COUNT = 0.00001

    for i in range(0, len(groundTruthList)):
        growth = groundTruthList[i]
        feeddata = feeddatalist[i]
        inputData = np.array(feeddata).reshape(1, KDAYS*5)
        currPred = sess.run(prediction, feed_dict={xs:inputData, keep_prob:1})
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
            TRADE_COUNT+=1.0
        
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
            
        if upPoss>=BUY_LINE:
            predictTotal += 1.0
            
        simtrade_netvalue_list.append(simtrade_netvalue)
        alpha_list.append(simtrade_netvalue/benchmark_netvalue-1)
        
        if i%30==0:
            percent = i/float(len(groundTruthList))
            print('Simulate Calc %0.2f%% ...' %(percent*100.0))
    
    print('%s Calc Finished!' %code)
    
    print('Accuracy : %0.2f%%' %(predictRight/predictTotal*100.0))
    print('Profit&loss Ratio : %0.2f%%' %((WIN_WEIGHT/LOSE_WEIGHT-1)*100))
    print('Hit Rate : %0.2f%%' %(HIT_COUNT/TOTAL_COUNT*100.0))
    print('Benchmark NetValue : %f' %(benchmark_netvalue))
    print('Simtrade NetValue : %f' %(simtrade_netvalue))
    print('Profit Per Day : %0.2f%%' %((simtrade_netvalue-1)/TRADE_COUNT*100.0))
    print('\n')
    
    str_code = 'CODE:%s' %(code)
    str_accuracy = '%0.2f%%' %(predictRight/predictTotal*100.0)
    str_proandlossRatio = '%0.2f%%' %((WIN_WEIGHT/LOSE_WEIGHT-1)*100)
    str_hitRate = '%0.2f%%' %(HIT_COUNT/TOTAL_COUNT*100.0)
    str_benchmarkValue = '%f' %(benchmark_netvalue)
    str_simtradeValue = '%f' %(simtrade_netvalue)
    str_profitPerDay = '%0.2f%%' %((simtrade_netvalue-1)/TRADE_COUNT*100.0)
    
    datalist = [str_code, str_accuracy, str_proandlossRatio, str_hitRate, str_benchmarkValue, str_simtradeValue, str_profitPerDay]
    
    table1 = pd.DataFrame()
    table1['股票代码'] = None
    table1['准确率'] = None
    table1['盈亏比'] = None
    table1['命中率'] = None
    table1['基准净值'] = None
    table1['交易净值'] = None
    table1['每日平均收益'] = None
    table1.loc[len(table1.index)] = datalist
    table1.to_csv(path_or_buf=OUTPUT_FILENAME, mode='a', index=False, header=False)