# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 16:02:38 2022

@author: 15942
"""
import pandas as pd
import numpy as np
import os
from os import path
from itertools import chain
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score, confusion_matrix
import datetime


# 使编码时不再按sort排序，而是自定义，用法：a=xx,b=list(label(a))
def label(array, list_class):
    labels = list_class
    return map(labels.index, array)


def load_dataset(path_pnt, list_train, list_test, list_class):
    cate = ["HA1", "MA1", "LA1", "HA2", "MA2", "LA2", "HA3", "MA3", "LA3", "HA4", "MA4", "LA4", "HA5", "MA5", "LA5",
            "HA6", "MA6", "LA6",
            "HB1", "MB1", "LB1", "HB2", "MB2", "LB2", "HB3", "MB3", "LB3", "HB4", "MB4", "LB4", "HB5", "MB5", "LB5",
            "HB6", "MB6", "LB6",
                          "HC1", "MC1", "LC1", "HC2", "MC2", "LC2", "HC3", "MC3", "LC3", "z2z", "unocc"]
    files = os.listdir(path_pnt)
    X_train = []
    X_train = pd.DataFrame(X_train)
    y_train_o = []
    X_test = []
    X_test = pd.DataFrame(X_test)
    y_test_o = []
    for file in files:
        r_realPath = path.join(path_pnt, file)
        # xlsx1 = pd.ExcelFile(r_realPath)
        # xlsx = xlsx1.parse(sheet_name=0, header=0, index_col=0)
        # x_in_xlsx = xlsx[
        #     ['1#', '2#', '3#', '4#', '5#', '6#', '7#', '8#', '9#', '10#', '11#', '12#', '13#', '14#', '15#']]
        xlsx = pd.read_csv(r_realPath)
        x_in_xlsx = xlsx[['1#', '2#', '3#', '4#', '6#', '7#', '8#', '9#', '10#', '11#', '12#', '13#', '14#', '15#']]

        # print(type(x_in_xlsx))
        # print(x_in_xlsx)
        y_in_xlsx = xlsx[['tag']]
        # print(y_in_xlsx)
        if files.index(file) in list_train:
            print("train"+str(file))
            # X_train.append(x_in_xlsx.values)
            X_train = pd.concat([X_train, x_in_xlsx])
            y_in_xlsx.loc[~y_in_xlsx["tag"].isin(cate), "tag"] = "z2z"

            y_train_o.append(y_in_xlsx.values)
        elif files.index(file) in list_test:
            print("test" + str(file))
            # X_test.append(x_in_xlsx.values)
            X_test = pd.concat([X_test, x_in_xlsx])
            y_in_xlsx.loc[~y_in_xlsx["tag"].isin(cate), "tag"] = "z2z"

            y_test_o.append(y_in_xlsx.values)

    # 获取一个LabelEncoder
    class_y = LabelEncoder()
    # encode class values as integers
    # class_y = class_y.fit(list_class)
    # y_train_o=class_y.transform(y_train_o.values)
    # y_test_o=class_y.transform(y_test_o.values)
    y_train_array = np.array(y_train_o)
    y_test_array = np.array(y_test_o)
    y_train_list = list(chain(*y_train_array))
    # print(y_train_o)
    tag_train = y_train_array.flatten()
    tag_train = tag_train.tolist()
    # print(tag_train)
    a = dict([[i, tag_train.count(i)] for i in tag_train])
    print(a)
    num_train = pd.DataFrame.from_dict(a, orient='index', columns=["train"])
    y_test_list = list(chain(*y_test_array))
    tag_test = y_test_array.flatten()
    tag_test = tag_test.tolist()
    b = dict([[i, tag_test.count(i)] for i in tag_test])
    print(b)
    num_test = pd.DataFrame.from_dict(b, orient='index', columns=["test"])
    df_num = pd.concat([num_train, num_test], axis=1)
    # print(df_num_z2z)
    df_num = df_num.reindex(list_class)
    # df_num.to_csv(r"D:\末日数牛\课题\内容\现场实测\办公室\model\matrix\class1c\num_train_test.csv")

    # y_train_o=class_y.transform(y_train_list)
    y_train_o = list(label(y_train_list, list_class))
    # print(y_train_o)
    # y_test_o=class_y.transform(y_test_list)
    y_test_o = list(label(y_test_list, list_class))
    # print(y_test_o)
    # print(y_test_o)
    # convert integers to dummy variables (one hot encoding)
    y_train = np_utils.to_categorical(y_train_o)
    y_test = np_utils.to_categorical(y_test_o)

    return X_train, y_train, X_test, y_test


# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # for i in range(n_in, 0, -1):
    # 不引入之前的时间
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    for column in agg.columns:
        agg[column].fillna("0", inplace=True)

    return agg


def cs_to_sl(dataset, n_time):
    print("######DATASET###########")
    # print(dataset)
    reframed = series_to_supervised(dataset, n_time, 0)
    # reframed.to_csv(r"D:\末日数牛\课题\内容\住宅\v1231\model\matrix\reframed.csv")
    # dataset=pd.concat(dataset)
    values = reframed.values
    print("######values###########")
    # print(values)

    values = values.astype('float32')

    # normalize features,现在数据倒是本来就在（0，1）了，应该没有影响，要是后面引入前x时刻预测值，应该得随着变的
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    print("----------scaler----------", scaler)
    print("----------reframed----------", reframed)

    # print(reframed.values.shape)
    # print(reframed.head())
    return reframed, scaler


def evaluate_model(trainX, trainy, testX, testy):
    trainX = trainX.astype(np.float32)
    trainy = trainy.astype(np.float32)
    testX = testX.astype(np.float32)
    testy = testy.astype(np.float32)
    verbose, epochs, batch_size = 1, 1, 1
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    print("-HERRRRRRRRRRRRRRRR!")
    print(trainX)
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    print("\n\n\n\n\nPredict Input Shape: ", testX.shape)
    print(testX)
    predy = model.predict(testX)
    predy = predy.argmax(axis=1)
    testy = testy.argmax(axis=1)

    tp_y1 = pd.DataFrame(testy)
    tp_y2 = pd.DataFrame(predy)
    tp_y = pd.concat([tp_y1, tp_y2], axis=1)
    tp_y.to_csv(r".\\" + str(n_timesteps) + "-" + str(
        batch_size) + "_" + datetime.datetime.now().strftime('%Y%m%d%H%M') + ".csv", header=['testy', 'predy'], )

    # 生成混淆矩阵
    conf_mat = confusion_matrix(testy, predy)
    conf_mat = pd.DataFrame(conf_mat)

    conf_mat.to_csv(r".\\" + str(n_timesteps) + "-" + str(
        batch_size) + "_" + datetime.datetime.now().strftime('%Y%m%d%H%M') + ".csv")
    # fig, ax = plt.subplots(figsize=(10,8))
    # sns.heatmap(conf_mat, annot=True, fmt='d',
    #             xticklabels=pd.DataFrame(testy).values, yticklabels=pd.DataFrame(testy).values)
    # plt.ylabel('实际结果',fontsize=18)
    # plt.xlabel('预测结果',fontsize=18)
    model.save('model.h5')
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10, prefix=''):
    # load data
    trainX, trainy, testX, testy = load_dataset(prefix)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


path_pnt = r".\data\p_n_t-30s"
# y_all=pd.read_csv(r"C:\Users\15942\OneDrive\文档\学位论文\内容\场景实测\211005-07\y\y_all.csv")
# y_all=pd.read_csv(r"D:\末日数牛\课题\内容\住宅\v1231\model\matrix\y\try.csv")
list_class = ["HA1", "MA1", "LA1", "HA2", "MA2", "LA2", "HA3", "MA3", "LA3", "HA4", "MA4", "LA4", "HA5", "MA5", "LA5",
              "HA6", "MA6", "LA6",
              "HB1", "MB1", "LB1", "HB2", "MB2", "LB2", "HB3", "MB3", "LB3", "HB4", "MB4", "LB4", "HB5", "MB5", "LB5",
              "HB6", "MB6", "LB6",
                            "HC1", "MC1", "LC1", "HC2", "MC2", "LC2", "HC3", "MC3", "LC3", "z2z", "unocc"]

list_train = [0, 2, 5, 6, 8, 10, 11, 12, 13, 16, 17, 18, 19, 21]
list_test = [1, 3, 4, 7, 9, 14, 15, 20, 22]

trainX, trainy, testX, testy = load_dataset(path_pnt, list_train, list_test, list_class)
print("#########trainX#########")
print(trainX)

# n_times=[1,5,10,20,30,40,50,60,120,300,600]
n_times = [10]
# n_times=[1,2]
accs = []
n_pir = 14
for n_time in n_times:
    print("---------开始" + str(n_time) + "s运算----------")
    trainX2, scaler1 = cs_to_sl(trainX, n_time)
    # trainX2.to_csv(r"D:\末日数牛\课题\内容\住宅\v1231\model\trainX2.csv")
    testX2, scaler2 = cs_to_sl(testX, n_time)
    # trainX2=np.array(trainX2)
    trainX2 = trainX2.values
    # trainX2.to_csv(r"D:\末日数牛\课题\内容\住宅\v1231\model\matrix\trainX2.csv")
    trainy = np.array(trainy)
    testX2 = testX2.values
    testy = np.array(testy)

    # reshape the data to satisfy the input acquirement of LSTM
    trainX2 = trainX2.reshape(trainX2.shape[0], n_time, n_pir)
    testX2 = testX2.reshape(testX2.shape[0], n_time, n_pir)
    trainy = trainy.reshape(trainy.shape[0], len(list_class))
    testy = testy.reshape(testy.shape[0], len(list_class))
    # print(trainX2.shape)
    # print(trainX2)
    print(trainX2.shape, trainy.shape, testX2.shape, testy.shape)
    acc = evaluate_model(trainX2, trainy, testX2, testy)
    accs.append(acc)
    print(acc)
    print("---------结束" + str(n_time) + "s运算----------")
n_times = pd.DataFrame(n_times)
accs = pd.DataFrame(accs)
n_times_accs = pd.concat([n_times, accs], axis=1)
n_times_accs.to_csv(r".\\" + str(list_test) + datetime.datetime.now().strftime(
    '%Y%m%d%H%M') + ".csv")
print("over")
