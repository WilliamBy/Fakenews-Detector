# 说明文档

### 项目框架和库版本

- Python 3.8.0
- scikit-learn 1.0.1
- numpy  1.21.2
- pandas 1.3.4
- joblib 1.1.0
- jieba 0.42.1
- 开发工具：Pycharm Professional 2021.2.2

### 目录结构及说明

```
root
 ├── CS1901_柏威良_U201914899_项目报告.docx
 ├── ML		//项目文件夹
 │   ├── dataset	//数据集
 │   │   ├── cn_stopwords.txt	//停用词表，预处理用
 │   │   ├── test.csv			//原始测试集
 │   │   ├── test.pkl			//预处理测试集
 │   │   ├── train.csv			//原始训练集
 │   │   └── train.pkl			//预处理训练集
 │   ├── fit.py					//训练模块
 │   ├── mlp_pred.txt			//预测结果，由predict.py产生
 │   ├── model					//保存模型的文件夹
 │   │   ├── mlp.model			//多层感知机模型
 │   │   ├── scaler.model		//数据标准化模型
 │   │   └── wv.model			//词向量模型
 │   ├── optimize.py			//自动参数选择模块
 │   ├── predict.py				//预测模块
 │   └── preprocess.py			//预处理模块
 └── 说明手册.md
```

### 使用流程

1. `preprocess.py` 会产生预处理数据集和词向量模型
2. `optimize.py` 会打印自动选取的参数，训练时间较长，项目中已经应用了参数因此这一步**可以不执行**
3. `fit.py` 训练，产生训练好的 mlp 模型
4. `predict.py` 预测测试集并将预测结果写到`mlp_pred.txt`文件中