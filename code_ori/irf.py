#coding=utf-8
import serial  # 导入模块
import datetime

try:
    # pre = 32768
    portx = "COM5"
    bps = 115200
    # 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
    timex = 10
    ser = serial.Serial(portx, bps, timeout=timex)

    i = datetime.datetime.now() #获取当前时间
    time_str = ("RTC_Set(%s,%s,%s,%s,%s,%s)\r\n" % (i.year, i.month, i.day, i.hour, i.minute, i.second+1)).encode()  #拼合数组
    print(time_str) #打印数组
    ser.write(time_str) #设置时间
    last_time = datetime.datetime.now()
    # 循环接收数据，此为死循环，可用线程实现
    while True:
        if ser.in_waiting:  
            str = ser.readline() 
            if (str == "exit"):  # 退出标志
                break
            else:
                print(str)              
    print("---------------")
    ser.close()  # 关闭串口


except Exception as e:
    print("---异常---：", e) 
