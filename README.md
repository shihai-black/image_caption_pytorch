# README

# 简介

基于pytorch，复现《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》论文

# 主要贡献

踩了一遍参考代码的坑，修复了预测时出现的bug。

TODO：增加预测的Beam_Search代码

# 运行

**数据预处理**

```python
python3 preprocess.py 
```

**主程序**

*训练*

```python
python3 run.py -bs 64 --cuda  --epochs 20
```

*预测*

```python
python3 run.py -bs 64 --cuda  --epochs 20 -p
```

# 结论

**数据集**：'Flicker8k'【https://www.kaggle.com/ming666/flicker8k-dataset?select=Flickr8k_text】

**结果**

| Beam Size | Validation BLEU-4 | Test BLEU-4 |
| --------- | ----------------- | ----------- |
| 1         | 0.185             | 0.196       |
| 3         | Todo              | Todo        |
| 5         | Todo              | Todo        |

# 参考

paper link：https://arxiv.org/pdf/1502.03044v3.pdf

reference code:https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

