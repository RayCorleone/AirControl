# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @version:   1.0
    @function:  后台运行该脚本，以实时获得数据
    @problem:   (1)如何连接外设
                (2)label倒数第二个是"z2z", 暂时无用 
"""

import time
import datetime
import numpy as np

########################################################
## 全局参数、类、函数设置
H_THRESHOLD = 1.34                  #高水平阈值
M_THRESHOLD = 0.77                  #中水平阈值
L_THRESHOLD = 0.05                  #低水平阈值

TRAIN_MEAN  = 14397.779677419356    #训练集MEAN
TRAIN_STD   = 19534.963617282596    #训练集STD
DEVICE_NUM  = 14    #15             #设备数量
DATA_NUM    = 10                    #输入模型的数据量(积攒的秒数)
LABLE_SIZE  = 47    #12             #label的个数
BATCH_SIZE  = 1     #20             #预测时使用的数据量

INTERVAL    = 20 * 1                #参数调节的时间间隔
SP_DIVIDER  = np.minimum(int(INTERVAL/3), 10)   #判断特殊事件使用的数据量大小
SAVE_FOLDER = './data/saved/'       #处理后的数据保存的路径
MODEL_NAME  = 'lstm.h5'             #使用的模型名
TEST_FILE   = 'test_lstm.txt'       #测试数据文件名(0,1序列文件)

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
def load_model_and_config(path = './model/'+MODEL_NAME):
    from keras.models import load_model
    print("-NOTICE: Loading Model & Config...")

    # 加载模型
    model = load_model(path)

    # 加载标签列表(可考虑从文件加载)
    label = Label() #P.S.倒数第二个标是"z2z",暂时不考虑
    label.all = ["HA1", "MA1", "LA1", "HA2", "MA2", "LA2", "HA3", "MA3", "LA3", "HA4", "MA4", "LA4", "HA5", "MA5", "LA5",
                 "HA6", "MA6", "LA6", "HB1", "MB1", "LB1", "HB2", "MB2", "LB2", "HB3", "MB3", "LB3", "HB4", "MB4", "LB4",
                 "HB5", "MB5", "LB5", "HB6", "MB6", "LB6", "HC1", "MC1", "LC1", "HC2", "MC2", "LC2", "HC3", "MC3", "LC3", "unocc", "unocc"]
                # ['HA1','HA2','HA3','LA1','LA2','LA3','LB1','LB3','MA1','MA2','MA3','unocc']
    label.ABC = ["A1", "A1", "A1", "A2", "A2", "A2", "A3", "A3", "A3", "A4", "A4", "A4", "A5", "A5", "A5",
                 "A6", "A6", "A6", "B1", "B1", "B1", "B2", "B2", "B2", "B3", "B3", "B3", "B4", "B4", "B4",
                 "B5", "B5", "B5", "B6", "B6", "B6", "C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3", "unocc", "unocc"]
                # ['A1','A2','A3','A1','A2','A3','B1','B3','A1','A2','A3','unocc']
    label.HML_str = ["H", "M", "L", "H", "M", "L", "H", "M", "L", "H", "M", "L", "H", "M", "L",
                     "H", "M", "L", "H", "M", "L", "H", "M", "L", "H", "M", "L", "H", "M", "L",
                     "H", "M", "L", "H", "M", "L", "H", "M", "L", "H", "M", "L", "H", "M", "L", "unocc", "unocc"]
                    # ['H','H','H','L','L','L','L','L','M','M','M','unocc']
    label.HML_int = [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0,
                     2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0,
                     2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 0]
                    # [2,2,2,0,0,0,0,0,1,1,1,0]

    return model, label

## 对数据预测(classes, t_label)
def predict(input, model):
    # 各个类的概率
    classes = model.predict(input, batch_size=BATCH_SIZE)
    classes = classes[0]

    # 概率最大的类
    t_label = np.argmax(classes, axis=0)

    return classes, t_label

## 特殊情况判断(move_flag, leave_flag, area)
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

device_in_data_ori = []                         #设备输入的数据(0,1列表)
device_in_data_list = []                        #设备输入的连续累积(十进制列表)
model_in_data_ori = []                          #模型输入的单个数据(列表)
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
file = open('./data/test/'+TEST_FILE,'r')

model, label = load_model_and_config()

device_in_stable_cnt = 0        #设备输入起步计数
device_in_stable_flag = False   #设备输入稳定标志
model_in_start_cnt = 0          #模型输入起步计数
model_in_stable_flag = False    #模型输入稳定标志
cumulate_cnt = 0                #间隔计数
cumulate_flag = False   #间隔稳定标志
interval_flag = False   #间隔标志

while True:
    time.sleep(0.5)
    str = file.readline() 
    if (str == "exit"):  # 退出标志
        print("\n-NOTICE: Device Closing...")
        break
    else:
        # 0.对设备直传的数据处理：字符串->0,1列表  (NOTICE:目前简化成过去30s数据的平均值)
        str_ori = str.replace('\n','').split(',')
        device_in_data_ori = [float(x) for x in str_ori]    # 设备输入平均值时
        # device_in_data_ori = [int(x) for x in str_ori]    # 设备输入0和1时
        np.save(SAVE_FOLDER+'device_in_data_ori.npy',np.array(device_in_data_ori))


        # 1.对设备直传数据进行运算并累积  (NOTICE:简化成过去30s数据的平均值后，这一步改变)
        # value = 0
        # for i in range(DEVICE_NUM):
        #     value = value + device_in_data_ori[i] * (2**(i+1))

        if device_in_stable_flag == False: #设备输入还没稳定，不足以构成单个模型输入
            device_in_data_list.append(device_in_data_ori)  #(value)
            device_in_stable_cnt = device_in_stable_cnt + 1
            if device_in_stable_cnt == DATA_NUM:
                print("\n-NOTICE: Device Input Data is Stable. Waiting For Batch Now...")
                device_in_stable_flag = True
        else: #设备输入已经稳定
            device_in_data_list = device_in_data_list[1:]
            device_in_data_list.append(device_in_data_ori)  #(value)

        
        # 2.传递累积DATA_NUM的一组数据
        if device_in_stable_flag:
            model_in_data_ori = device_in_data_list
            np.save(SAVE_FOLDER+'model_in_data_ori.npy',np.array(model_in_data_ori))
        else:
            continue

        

        # 3.累积获得输入模型的数据列表
        if model_in_stable_flag == False: #还没稳定
            in_data_list = in_data_list + model_in_data_ori
            model_in_start_cnt = model_in_start_cnt + 1
            if model_in_start_cnt == BATCH_SIZE:
                print("\n-NOTICE: Batch Input Data is Stable. Predicting Now...")
                model_in_stable_flag = True
        else:   #已经稳定
            in_data_list = in_data_list[DATA_NUM:] + model_in_data_ori


        # 4.将数据列表标准化并转化为3维数组     (NOTICE:LSTM是3维数组，Keras是4维数组)
        if model_in_stable_flag:
            in_data = np.array(in_data_list).reshape((BATCH_SIZE, DATA_NUM, DEVICE_NUM))
            # in_data = (in_data - TRAIN_MEAN)/(TRAIN_STD)
            # in_data = np.array(in_data_list).reshape((BATCH_SIZE, DATA_NUM,1,1))
            # in_data = (in_data - TRAIN_MEAN)/(TRAIN_STD)
        else:
            continue
        np.save(SAVE_FOLDER+'in_data.npy',in_data.reshape((DATA_NUM,DEVICE_NUM)))
        # np.save(SAVE_FOLDER+'in_data.npy',in_data.reshape((BATCH_SIZE, DATA_NUM)))


        # 5.通过模型进行计算
        classes, label_index = predict(in_data, model)
        np.save(SAVE_FOLDER+'classes.npy',classes)


        # 6.模型计算结果的连续累积
        if cumulate_flag == False: #还没稳定
            label_index_smooth.append(label_index)
            label_HML_smooth.append(label.HML_int[label_index])
            np.save(SAVE_FOLDER+'label_HML_smooth.npy',np.array(label_HML_smooth))
            np.save(SAVE_FOLDER+'label_index_smooth.npy',np.array(label_index_smooth))

            cumulate_cnt = cumulate_cnt + 1
            if cumulate_cnt >= INTERVAL:
                print("\n-NOTICE: Predicting Data is Now Stable.")
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
            np.save(SAVE_FOLDER+'label_HML_smooth.npy',np.array(label_HML_smooth))
            np.save(SAVE_FOLDER+'label_index_smooth.npy',np.array(label_index_smooth))

            cumulate_cnt = cumulate_cnt + 1
            if cumulate_cnt >= INTERVAL:
                interval_flag = True
                cumulate_cnt = cumulate_cnt - INTERVAL


        # 7.到达间隔时间后的数据截取和计算
        if interval_flag == True:
            curr_time = datetime.datetime.now()
            timestamp = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
            print("\n-NOTICE: Reached Interval at", timestamp)

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
            
            np.save(SAVE_FOLDER+'label_HML_capture.npy',np.array(label_HML_capture))
            np.save(SAVE_FOLDER+'label_index_capture.npy',np.array(label_index_capture))

        else:   #未到达间隔时间时, 判断特殊情况
            sudden_move_flag, sudden_leave_flag, sudden_area = special_events(label_HML_smooth, label)
        

        # 8.特殊情况/到达间隔时的参数生成
        if sudden_move_flag:
            print("\n-TESTING: Sudden Movement...")
            air_configs = AirConfig(1,1,sudden_area,'H')
            sudden_move_flag = False
            print('--Current Configs:', air_configs.angle1,air_configs.angle2,air_configs.temp,air_configs.speed)
        elif sudden_leave_flag:
            print("\n-TESTING: Leave The Room...")
            air_configs = AirConfig(1,0,'A1','L')
            sudden_leave_flag = False
            print('--Current Configs:', air_configs.angle1,air_configs.angle2,air_configs.temp,air_configs.speed)
        elif interval_flag:
            print("\n-NOTICE: Geting Configs...")
            air_configs = AirConfig(0,1,label.ABC[last_index],HML_average_str)
            interval_flag = False
            print('--Current Configs:', air_configs.angle1,air_configs.angle2,air_configs.temp,air_configs.speed)
