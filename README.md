# AirControl
> 节能减排竞赛：基于 PIR 的空调控制系统设计
>
> **P.S. 更多项目材料，请联系邮箱 rayhuc@163.com，并备注说明来意 (请注意基本的礼貌，谢谢)**

<br/>

### 基本信息

- 作者：Ray
- 时间：2022-06-29
- 备注：由于文件大小限制，删去了所有的数据和训练好的模型，需要请联系邮箱；

<br/>

### 文件目录

- **`.backup` 文件夹**：历史代码文件的备份
  - `connect` 文件夹：设备连接代码
    - irf.py：设备直接连接代码文件
  - `test` 文件夹：测试数据生成代码
    - make_data.py：第一随机测试数据生成
  - `train` 文件夹：模型训练代码
    - fcn_train.py：fcn 模型训练
    - lstm_train.py：lstm 模型训练
  - process_1st.py：第一版完成的完整的计算脚本

- **`data` 文件夹**：数据文件
  - `fcn_train_data` 文件夹：fcn 模型训练数据
    - ( ..... )
  - `lstm_train_data` 文件夹：lstm 模型训练数据
    - ( ..... )
  - `saved` 文件夹：中间运算数据
    - ( ..... )
  - `test` 文件夹：脚本测试数据
    - ( ..... )

- **`flask` 文件夹**：展示程序的网页端 (待上线)
  - ( ..... )

- **`matplot` 文件夹**：基于 matplotlib 的展示程序
  - draw.py：绘图展示脚本

- **`model` 文件夹**：部署的训练模型
  - keras.h5：简易 fcn 模型
  - keras_2000.h5：训练 2000轮的 fcn 模型
  - lstm.h5：lstm 模型

- **`train` 文件夹**：模型训练代码 (未在 pipenv 中安装全部库)
  - fcn_train.py：fcn 模型训练
  - lstm_train.py：lstm 模型训练

- **Pipfile**：Pipenv 管理的环境文件
- **Pipfile.lock**：Pipenv 管理的环境文件 lock
- **process.py**：数据运算脚本
- **README.md**：代码说明文件 (此文件)

<br/>

###  运行环境

1. 使用 Python3.9，pipenv 环境管理

2. 代码：

   ```bash
   cd <current_directory>
   pipenv install
   pipenv shell
   ```
