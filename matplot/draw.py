# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @version:   1.0
    @function:  运行该脚本以实时绘制图像
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


########################################################
## 全局参数、类、函数设置
H_THRESHOLD = 1.34
M_THRESHOLD = 0.77
L_THRESHOLD = 0.05
DEVICE_NUM  = 14
DATA_NUM    = 10
LABLE_SIZE  = 47
BATCH_SIZE  = 1
INTERVAL    = 20 * 1
TIME_UPDATE = 5    #小于INTERVAL
SP_DIVIDER  = np.minimum(int(INTERVAL/3), 10)
SAVE_FOLDER = './data/saved/'

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

## 载入参数(label)
def load_config():
    print("-NOTICE: Loading Config...")

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

    return label

## 实时数据获取标签(t_label)
def get_label(input):
    t_label = np.argmax(input, axis=0)
    return t_label

## 间断数据获取信息(last_index, HML_average, HML_average_str)
def get_interval_info(label_index_capture, label_HML_capture):
    last_index = label_index_capture[-1]
    HML_average = np.average(label_HML_capture)
    HML_average_str =' '

    if HML_average >= H_THRESHOLD:
        HML_average_str = 'H'
    elif HML_average >= M_THRESHOLD:
        HML_average_str = 'M'
    elif HML_average >= L_THRESHOLD:
        HML_average_str = 'L'
    else:
        HML_average_str = 'unocc'
    
    return last_index, HML_average, HML_average_str

## 特殊情况判断(move_flag, leave_flag, area)
def special_events(index_list, label):
    move_flag = False
    leave_flag = False
    area = 'A1'
    area_index = index_list[-SP_DIVIDER]
    average_past = np.average(index_list[-2*SP_DIVIDER:-SP_DIVIDER])
    average_now  = np.average(index_list[-SP_DIVIDER:])

    if average_past>M_THRESHOLD and average_now<L_THRESHOLD:    #离开
        leave_flag = True
    elif average_past<L_THRESHOLD and average_now>M_THRESHOLD:  #进入
        move_flag = True
        area = label.ABC[area_index]
        
    return move_flag, leave_flag, area


########################################################
## 全局数据声明
label = load_config()
max_y_scatter = 0
curr_time = datetime.datetime.now()


########################################################
## 绘图部分
plt.style.use('ggplot')
fig = plt.figure()
gs = GridSpec(2,6,figure=fig)
# gs = GridSpec(2,2,figure=fig)


# ax1散点图: 设备实时输入数据
ax1 = fig.add_subplot(gs[0,0:2])
scats = []
vlns = []
scat = ax1.scatter([], [], label='Value', c='r', marker='o', ls ='-')
vln = ax1.axvline(0,0,0, c='b', ls='-')
scats.append(scat)
vlns.append(vln)
# for i in range(1, DATA_NUM):
for i in range(1, DEVICE_NUM):
    scat = ax1.scatter([], [], c = 'r', marker = 'o', ls = '-')
    vln = ax1.axvline(0,0,0, c='b', ls='-')
    scats.append(scat)
    vlns.append(vln)


# ax2伪柱状图: 模型实时预测出的概率(图例是预测标签)
ax2 = fig.add_subplot(gs[0,2:6])
bars = []
for i in range(0,LABLE_SIZE):
    bar = ax2.axvline(0,0,0, c='c', ls='-')
    bars.append(bar)


# ax3折线图: 实时积累的强度累积图(图例是是否有特殊事件)
ax3 = fig.add_subplot(gs[1,0:3])
ln1, = ax3.plot([], [], c = 'm', ls = '-')


# ax4折线图: 阶段截取的强度累积图(图例是时间戳 + 综合判断 + 调节参数)
ax4 = fig.add_subplot(gs[1,3:6])
ln2, = ax4.plot([], [], color=[0.3,0.5,0.7], marker = '.', ls = '-')


def init():
    global label

    # ax1散点图
    ax1.legend(loc='upper right',fontsize='small')
    # ax1.set_title("Input Slices", fontsize='large')
    # ax1.set_xlabel("Time(s)", fontsize='medium')
    # ax1.set_ylabel("Merged Value", fontsize='medium')
    # ax1.set_xticks(np.arange(start=0, stop=DATA_NUM, step=1))
    # ax1.set_xlim((-0.5, DATA_NUM-0.5))
    ax1.set_title("Device Input Arvage", fontsize='large')
    ax1.set_xlabel("Device Number", fontsize='medium')
    ax1.set_ylabel("Value", fontsize='medium')
    ax1.set_xticks(np.arange(start=0, stop=DEVICE_NUM, step=1))
    ax1.set_xlim((-0.5, DEVICE_NUM-0.5))

    # ax2伪柱状图
    ax2.legend(['Waiting'],loc='upper right',fontsize='large',markerscale=0,handlelength=0,labelcolor='r')
    ax2.set_title("Classification", fontsize='large')
    ax2.set_xlabel("Label Class", fontsize='medium')
    ax2.set_ylabel("Probability", fontsize='medium')
    ax2.set_xticks(np.arange(start=0, stop=LABLE_SIZE, step=1), label.all,fontsize=5)
    ax2.set_xlim((-0.5, LABLE_SIZE-0.5))
    ax2.set_yticks(np.arange(start=0, stop=1.1, step=0.1))
    ax2.set_ylim((-0.1, 1.1))

    # ax3折线图
    ax3.legend(['Waiting'],loc='upper right',fontsize='large',markerscale=0,handlelength=0,labelcolor='r')
    ax3.set_title("Realtime Intensity", fontsize='large')
    ax3.set_xlabel("Time(s)", fontsize='medium')
    ax3.set_ylabel("Intensity", fontsize='medium')
    ax3.set_xticks(np.arange(start=0, stop=INTERVAL+1, step=int(INTERVAL/20)))
    ax3.set_xlim((-INTERVAL*0.02, INTERVAL*1.01))
    ax3.set_yticks(np.arange(start=0, stop=2.1, step=1),['L','M','H'])
    ax3.set_ylim((-0.1, 2.1))

    # ax4折线图
    ax4.legend(['Waiting'],loc='upper right',fontsize='large',markerscale=0,handlelength=0,labelcolor='r')
    ax4.set_title("Captured Intensity", fontsize='large')
    ax4.set_xlabel("Time(s)", fontsize='medium')
    ax4.set_ylabel("Intensity", fontsize='medium')
    ax4.set_xticks(np.arange(start=0, stop=INTERVAL+1, step=int(INTERVAL/20)))
    ax4.set_xlim((-INTERVAL*0.02, INTERVAL*1.01))
    ax4.set_yticks(np.arange(start=0, stop=2.1, step=1),['L','M','H'])
    ax4.set_ylim((-0.1, 2.1))

    return scats, vlns, bars, ln1, ln2
 

def update(i):
    global label
    global curr_time
    global max_y_scatter

    try:
        ##################################
        # 读入数据
        device_in_data_ori = np.load(SAVE_FOLDER+'device_in_data_ori.npy', allow_pickle=True)
        # model_in_data_ori = np.load(SAVE_FOLDER+'model_in_data_ori.npy', allow_pickle=True)
        classes = np.load(SAVE_FOLDER+'classes.npy')
        label_HML_smooth = np.load(SAVE_FOLDER+'label_HML_smooth.npy')

        try:
            label_HML_capture = np.load(SAVE_FOLDER+'label_HML_capture.npy')
            label_index_capture = np.load(SAVE_FOLDER+'label_index_capture.npy')
            interval_flag = True
        except:
            interval_flag = False


        ##################################
        # 重新绘制散点图ax1
        ax1_ydata = device_in_data_ori
        # ax1_ydata = model_in_data_ori
        max_y_scatter = np.maximum(np.max(ax1_ydata), max_y_scatter)
        ax1.set_ylim((-max_y_scatter*0.02, max_y_scatter*1.02))
        # for i in range(0,DATA_NUM):
        for i in range(0,DEVICE_NUM):
            scats[i].set_offsets((i, ax1_ydata[i]))
            vlns[i].set_data([i,i],[0, ax1_ydata[i]/max_y_scatter/1.04])


        ##################################
        # 重新绘制柱状图ax2 (更新图例：预测标签)
        ax2_ydata = classes
        for i in range(0,LABLE_SIZE):
            bars[i].set_data([i,i],[0, (ax2_ydata[i]*10+0.5)/12])
            bars[i].set_lw(5)

        t_label = get_label(ax2_ydata)
        ax2_label = label.all[t_label]
        ax2.legend([ax2_label],loc='upper right',fontsize='large',markerscale=0,handlelength=0,labelcolor='r')


        ##################################
        # 重新绘制折线图ax3 (更新图例：是否有特殊事件)
        ax3_xdata = range(0,len(label_HML_smooth))
        ax3_ydata = label_HML_smooth
        ln1.set_data(ax3_xdata, ax3_ydata)

        ax3_label = 'None'
        sudden_move_flag, sudden_leave_flag, sudden_area = special_events(label_HML_smooth, label)
        if sudden_move_flag or sudden_leave_flag:
            np.save('s_device_in_data_ori.npy',np.array(device_in_data_ori))
            np.save('s_classes.npy',np.array(classes))
            np.save('s_label_HML_smooth.npy',np.array(label_HML_smooth))
        if sudden_move_flag:
            ax3_label = sudden_area + 'Entering'
        elif sudden_leave_flag:
            ax3_label = sudden_area + 'Leaving'
        ax3.legend([ax3_label],loc='upper right',fontsize='large',markerscale=0,handlelength=0,labelcolor='r')


        ##################################
        # 重新绘制折线图ax4 (更新图例：时间戳 + 综合判断 + 调节参数)
        if interval_flag and ((datetime.datetime.now()-curr_time).seconds > TIME_UPDATE):
            ax4_xdata = range(0,INTERVAL)
            ax4_ydata = label_HML_capture
            ln2.set_data(ax4_xdata, ax4_ydata)

            curr_time = datetime.datetime.now()
            ax4_label_data = label_index_capture
            timestamp = datetime.datetime.strftime(curr_time, '%H:%M:%S')
            last_index, HML_average, HML_average_str = get_interval_info(ax4_label_data, ax4_ydata)
            air_configs = AirConfig(0,0,'A1','L')
            if sudden_move_flag:
                air_configs = AirConfig(1,1,sudden_area,'H')
                sudden_move_flag = False
            elif sudden_leave_flag:
                air_configs = AirConfig(1,0,'A1','L')
                sudden_leave_flag = False
            elif interval_flag:
                air_configs = AirConfig(0,1,label.ABC[last_index],HML_average_str)
                interval_flag = False

            ax4_label = 'Time: '+timestamp+'\n'+\
                        'HML_Avg: {0:.3f}\n'.format(HML_average)+\
                        'Status: '+label.ABC[last_index]+'('+HML_average_str+')\n'+\
                        'Angle1: '+str(air_configs.angle1)+'\n'+\
                        'Angle2: '+str(air_configs.angle2)+'\n'+\
                        'Temperature: '+str(air_configs.temp)+'\n'+\
                        'Wind Speed: '+str(air_configs.speed)
            ax4.legend([ax4_label],loc='upper right',fontsize='medium',markerscale=0,handlelength=0,labelcolor=[0.3,0.5,0.9])

        return scats, vlns, bars, ln1, ln2

    except:
        print('Fail to open...')

ani = FuncAnimation(fig, update, interval = 500, init_func=init)
plt.tight_layout()
plt.show()