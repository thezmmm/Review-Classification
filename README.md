## Directory
- RNN
  - data
    - curTest
    
      此次测试所使用的数据
    
    - curTrain
    
      此次训练所使用的数据
    
    - realResult
    
      此次测试所使用数据的真实结果
    
    - testResult
    
      模型预测curTest生成的结果
    
    - train
    
      训练所使用数据
    
  - model
  
    - gloveModel
  
      词向量模型
  
    - RNN
  
      此文件夹下准备了三个已经训练好的模型文件
  
    - RNN.pkl
  
      使用curTrain训练得到的模型
  
  - datahandle
  
    数据处理
  
  - Initialize
  
    使用train随机生成curTest和curTrain
  
  - model
  
    模型训练
  
  - test
  
    模型预测

## Parameter

- N 

  将数据分成N份进行训练

- RNNClassificationModel

  - lr

    learning rate 学习速率

  - epoches

    训练周期

  - x

    每个训练周期用的数据量

    ```py
    for epoch in range(self.epoches):
        for i in range(x):
    ```

- SeqRNN

  - dropout

    将神经元丢弃的概率，即不进行训练，避免过拟合

- test

  ```python
  for i in range(x):
      onetest = testData[math.floor(i / x * lenOftest):math.floor((i + 1) / x * lenOftest)]
  ```

### 参数调整

1. 每次调参进行3-4次训练，得到平均正确率
2. 对数据进行初始化，重新生成训练集和测试集
3. 进行模型训练
4. 测试得到正确率

## 运行说明

操作系统：Windows10系统，Python版本3.9.7，在Pycharm下的终端执行。

依赖的包(除torch)，joblib,matplotlib,gensim,math,pandas,sklearn(ver1.1.3),os,sys,shutil

依次用python命令运行以下文件：
运行`initialize.py` 重新划分一次训练数据和测试数据；
运行`datahandle.py` 导入词向量模型，对数据进行处理；
运行`model.py` 训练模型,并在`model`目录下生成模型文件`RNN.pkl`；
运行`test.py` 评估模型性能。

## Test

测试需要运行 `test.py` 文件进行测试

函数`test()`为导入模型，参数为模型的相对路径，生成预测结果并保存在`data`目录下的`testResult.csv`中

函数`accuracy()`比对真实结果和预测结果，从而评估模型性能
