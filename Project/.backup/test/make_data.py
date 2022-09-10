import numpy as np
import pandas as pd 
import os 
import sys 
from pathlib import Path

def make_data(src_fpath, drc_fpath):
    # fpath = "D:/fqh_Workspace/2022-AutoAirCondition/data/home/p_n_t-xlsx-re/200208.xlsx"
    df = pd.read_excel(src_fpath)
    data_mat = df.to_numpy()

    # 将15个传感器的输出整合为一个16bit的数 
    value_list = []
    for line in data_mat:
        value = 0
        for i in range(15):
            if line[i+1] == 1:
                value += 2**(i+1)
        value_list.append(value)
    
    # 拼接成data_value_mat
    value_mat = np.asarray(value_list)
    data_value_mat = np.hstack((data_mat, value_mat.reshape(-1,1)))

    line_duration = 10    #每行数据的持续时间为 10s
    f = open(drc_fpath/os.path.basename(src_fpath), "w+")
    max_line = data_value_mat.shape[0]
    valid_line_cnt = 0
    tag_invalid_line_cnt = 0
    val_invalid_line_cnt = 0
    # 遍历整个数据集，生成所需的数据格式
    for ridx, line in enumerate(data_value_mat):
        ### init
        make_line = []        #一行处理好的数据
        vote_tag  = {}        #该行数据的标签评选
        ### 往下遍历10行, 整理数据
        overflow_flag = 0
        for i in range(line_duration):
            if ridx+i >= max_line:
                overflow_flag = 1
                break
            # make value series
            make_line.append(data_value_mat[ridx+i][17])
            # tag vote
            tag = data_value_mat[ridx+i][16]
            if tag in vote_tag:
                vote_tag[tag] += 1
            else:
                vote_tag[tag] = 1
        if overflow_flag:
            break # 余下数据不足line_duration
        
        ### 计算tag 
        max_tname =  list(vote_tag.keys())[0]
        max_vnum  =  vote_tag[max_tname]
        sum_vnum  =  0
        for tag_item in vote_tag.items():
            sum_vnum += tag_item[1]
            if tag_item[1] > max_vnum :
                max_vnum = tag_item[1]
                max_tname = tag_item[0]
        
        ### 检查数据的有效性：1.tag 2.val
        if float(max_vnum) / float(sum_vnum) < 0.8:
            tag_invalid_line_cnt+=1
            continue # 数据无效
        allzero = 1
        for v in make_line:
            if v:
                allzero = 0
                break
        if allzero:
            val_invalid_line_cnt+=1
            continue # 数据无效
        
        ### 将有效的数据写入文件
        f.write(str(max_tname))
        for v in make_line:
            f.write(","+str(v))
        f.write("\n")
        valid_line_cnt+=1
    f.close()
    print("{}:over. [ok:{} tag_drop:{} val_drop:{}]".format(os.path.basename(src_fpath), 
        valid_line_cnt, tag_invalid_line_cnt, val_invalid_line_cnt))

if __name__=='__main__':
    src_dir_path = Path(sys.argv[1])
    drc_dir_path = Path(sys.argv[2])
    flist = os.listdir(src_dir_path)
    for src_file in flist:
        make_data(src_dir_path/src_file, drc_dir_path)

