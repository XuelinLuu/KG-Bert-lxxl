#### KG-Bert-lxxl

---
- 本项目是通过Bert模型，使用数据集中数据来构建知识图谱
- 主要思想
    - 将head和tail作为上下句输入BERT中，将relation作为标签进行fine-tune，最终得到模型参数
    - 读取所有的实体分别作为head和tail，预测其所关系
    - 由于数据量巨大，且处理困难，且电脑硬件条件原因，分别将训练集和实体集进行了缩减，有能力可以将所有数据进行训练
    - 训练集恢复：删掉train.tsv，将train-副本.tsv重命名为train.tsv
    - 实体集恢复：将datasets.py中，KGPredictProcessor类中，_get_all_entities方法中的限制条件删除

---
#### 项目介绍
- datasets 中保存有所有的数据
- models 中保存训练之后的模型
- predictions/kg.tsv 中保存所有的三元组
- src 中保存项目

---
#### 运行
- 训练：run_train.py
- 预测：run_predict.py