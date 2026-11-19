# Breast Cancer Prediction From Scratch

课程项目：使用 C 语言从零实现基于梯度下降的乳腺癌数据二分类实验。

## 内容

- 30 维特征的数据读取与标准化
- 训练集 / 测试集划分
- 梯度下降优化
- 训练损失记录与预测结果输出

## 数据集

代码面向 Wisconsin Diagnostic Breast Cancer（WDBC）格式的数据集。出于数据来源与复现边界考虑，本仓库不直接分发原始 CSV；请从公开来源获取后放到项目根目录，并命名为 `breast_cancer.csv`。

## 构建与运行

```bash
gcc -O2 -std=c11 src/breast_cancer_lr.c -o breast_cancer_lr -lm
./breast_cancer_lr
```

Windows 下可将输出程序名改为 `breast_cancer_lr.exe`。

## 说明

本项目用于算法与 C 语言课程学习，不应用于真实医疗诊断。
