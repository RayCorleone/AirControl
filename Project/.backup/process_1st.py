# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @version:   1.0
    @problem:   (1)如何连接外设 
"""

import datetime
import numpy as np

########################################################
## 全局参数、类、函数设置
H_THRESHOLD = 1.34
M_THRESHOLD = 0.77
L_THRESHOLD = 0.05
TRAIN_MEAN  = 14397.779677419356
TRAIN_STD   = 19534.963617282596
DATA_NUM    = 10
LABLE_SIZE  = 12
BATCH_SIZE  = 20
INTERVAL    = 60 * 1
SP_DIVIDER  = np.minimum(int(INTERVAL/3), 20)

## 标签类
class Label():
    all = []
    ABC = []
    HML_str = []
    HML_int = []

    # 根据all标签设置其他标签
    def set_HML_ABC(self):
        pass

## 调整参数类
class AirConfig():
    area_dic = { # 地点: [θ,β,speed]
        'A1': [27.05,   41.08,  2.20],
        'A2': [41.64,   51.86,  1.88],
        'A3': [73.75,   61.48,  1.71],
        'A4': [118.45,  59.32,  1.74],
        'A5': [143.98,  48.43,  1.96],
        'A6': [150.50,  37.07,  2.37],
        'B1': [4.86,    44.28,  2.08],
        'B2': [8.43,    59.32,  1.74],
        'B3': [29.75,   80.06,  1.55],
        'B4': [162.91,  73.53,  1.58],
        'B5': [173.10,  54.15,  1.83],
        'B6': [175.70,  40.88,  2.21],
        'C1': [318.36,  51.86,  1.88],
        'C2': [249.44,  60.88,  1.71],
        'C3': [208.06,  42.05,  2.16],
        'unocc': [0, 0, 0]
    }

    HML_dic = { # HML: temp
        'H': 22,
        'M': 25,
        'L': 27,
        'unocc': 27
    }

    # special:是否特殊  ena:是否使能  area:地点  HML:强度
    def __init__(self, special, ena, area, HML):
        if not ena: #不使能(离开)
            self.ena = False
            self.angle1 = 0
            self.angle2 = 0
            self.temp = 27
            self.speed = 0
        else:   #普通情况
            #设置使能
            self.ena = True
            #设置温度
            self.temp = self.HML_dic[HML]
            #设置风速、角度
            self.angle1 = self.area_dic[area][0]
            self.angle2 = self.area_dic[area][1]
            self.speed  = self.area_dic[area][2]
            #突然出门, 温度不一样
            if special:
                self.temp = 23

## 载入模型和参数(model, label)
def load_model_and_config(path = './model/keras_try.h5'):
    from keras.models import load_model
    print("-NOTICE: Loading Model & Config...")

    # 加载模型
    model = load_model(path)

    # 加载标签列表(可考虑从文件加载)
    label = Label()
    label.all = ['HA1','HA2','HA3','LA1','LA2','LA3','LB1','LB3','MA1','MA2','MA3','unocc']
    label.ABC = ['A1','A2','A3','A1','A2','A3','B1','B3','A1','A2','A3','unocc']
    label.HML_str = ['H','H','H','L','L','L','L','L','M','M','M','unocc']
    label.HML_int = [2,2,2,0,0,0,0,0,1,1,1,0]

    return model, label

## 对数据预测(classes, t_label)
def predict(input, model):
    # 各个类的概率
    classes = model.predict(input, batch_size=BATCH_SIZE)
    classes = classes[0]

    # 概率最大的类
    t_label = np.argmax(classes, axis=0)

    return classes, t_label

## 特殊情况判断(move_flag, leave_flag)
def special_events(index_list, label):
    move_flag = False
    leave_flag = False
    area = 'A1'
    area_index = index_list[-SP_DIVIDER]
    average_past = np.average(index_list[-2*SP_DIVIDER:-SP_DIVIDER])
    average_now  = np.average(index_list[-SP_DIVIDER:])

    if average_past>H_THRESHOLD and average_now<L_THRESHOLD:    #离开
        leave_flag = True
    elif average_past<L_THRESHOLD and average_now>H_THRESHOLD:  #进入
        move_flag = True
        area = label.ABC[area_index]
        
    return move_flag, leave_flag, area


########################################################
## 提前声明重要数据
curr_time = datetime.datetime.now()     #当前时间
air_configs = AirConfig(0,0,'A1','L')   #调节参数

in_data_ori = []                                #设备输入的数据(列表)
in_data_list = []                               #输入模型的数据(列表)
in_data = np.zeros((BATCH_SIZE, DATA_NUM,1,1))  #输入模型的数据(数组)

label = Label()                     #全局标签
label_index = 0                     #模型预测返回的标签(一秒)
classes = np.zeros((LABLE_SIZE))    #模型预测的各个类的概率(数组)

label_index_smooth = []     #模型预测累积的标签(持续)
label_HML_smooth = []       #模型预测累积的强度(持续)

last_index = 0              #最后一个标签(INTERVAL点)
label_index_capture = []    #模型预测累积的标签(INTERVAL点)
label_HML_capture = []      #模型预测累积的强度(INTERVAL点)
HML_average = 0             #平均强度(INTERVAL点)
HML_average_str = 'L'       #平均强度(INTERVAL点)

sudden_area = 'A1'          #突然动作区域
sudden_move_flag = False    #突然动作标志
sudden_leave_flag = False   #突然离开标志


########################################################
## 虚拟数据运行
file = open("./data/test.txt",'r')

model, label = load_model_and_config()

in_start_cnt = 0        #起步计数
in_stable_flag = False  #起步稳定标志
cumulate_cnt = 0        #间隔计数
cumulate_flag = False   #间隔稳定标志
interval_flag = False   #间隔标志

while True:  
    str = file.readline() 
    if (str == "exit"):  # 退出标志
        print("-NOTICE: Device Closing...")
        break
    else:
        # 1.对设备直传的数据处理：字符串->列表  (待定)
        str_ori = str.replace('\n','').split(',')
        in_data_ori = [int(x) for x in str_ori]


        # 2.累积获得输入模型的数据列表
        if in_stable_flag == False: #还没稳定
            in_data_list = in_data_list + in_data_ori
            in_start_cnt = in_start_cnt + 1
            if in_start_cnt == BATCH_SIZE:
                print("-NOTICE: Input Data is Stable. Predicting Now...")
                in_stable_flag = True
        else:   #已经稳定
            in_data_list = in_data_list[DATA_NUM:] + in_data_ori


        # 3.将数据列表标准化并转化为4维数组
        if in_stable_flag:
            in_data = np.array(in_data_list).reshape((BATCH_SIZE, DATA_NUM,1,1))
            in_data = (in_data - TRAIN_MEAN)/(TRAIN_STD)
        else:
            continue


        # 4.通过模型进行计算
        classes, label_index = predict(in_data, model)


        # 5.模型计算结果的连续累积
        if cumulate_flag == False: #还没稳定
            label_index_smooth.append(label_index)
            label_HML_smooth.append(label.HML_int[label_index])

            cumulate_cnt = cumulate_cnt + 1
            if cumulate_cnt >= INTERVAL:
                print("-NOTICE: Predicting Data is Now Stable.")
                cumulate_flag = True
                interval_flag = True
                cumulate_cnt = cumulate_cnt - INTERVAL
            else:
                continue
        else:   #已经稳定
            label_index_smooth = label_index_smooth[1:]
            label_HML_smooth = label_HML_smooth[1:]
            label_index_smooth.append(label_index)
            label_HML_smooth.append(label.HML_int[label_index])
            
            cumulate_cnt = cumulate_cnt + 1
            if cumulate_cnt >= INTERVAL:
                interval_flag = True
                cumulate_cnt = cumulate_cnt - INTERVAL
        

        # 6.到达间隔时间后的数据截取和计算
        if interval_flag == True:
            curr_time = datetime.datetime.now()
            timestamp = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
            print("-NOTICE: Reached Interval at", timestamp)

            label_HML_capture = label_HML_smooth
            label_index_capture = label_index_smooth

            last_index = label_index_capture[-1]
            HML_average = np.average(label_HML_capture)

            if HML_average >= H_THRESHOLD:
                HML_average_str = 'H'
            elif HML_average >= M_THRESHOLD:
                HML_average_str = 'M'
            elif HML_average >= L_THRESHOLD:
                HML_average_str = 'L'
            else:
                HML_average_str = 'unocc'
        else:   #未到达间隔时间时, 判断特殊情况
            sudden_move_flag, sudden_leave_flag, sudden_area = special_events(label_HML_smooth, label)
        

        # 7.特殊情况/到达间隔时的参数生成
        if sudden_move_flag:
            print("-TESTING: Sudden Movement...")
            air_configs = AirConfig(1,1,sudden_area,'H')
            sudden_move_flag = False
        elif sudden_leave_flag:
            print("-TESTING: Leave The Room...")
            air_configs = AirConfig(1,0,'A1','L')
            sudden_leave_flag = False
        elif interval_flag:
            print("-NOTICE: Geting Configs...")
            air_configs = AirConfig(0,1,label.ABC[last_index],HML_average_str)
            interval_flag = False
