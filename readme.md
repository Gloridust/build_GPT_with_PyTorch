# 从零实现GPT模型

这是一个基于PyTorch从零实现的GPT(Generative Pre-trained Transformer)模型。该实现主要用于教学目的,帮助理解Transformer解码器的核心原理。

## 1. 模型原理

### 1.1 整体架构

GPT模型是一个基于Transformer解码器的自回归语言模型。其核心组件包括:

- 多头自注意力层(Multi-Head Self-Attention)
- 前馈神经网络层(Feed Forward Network)
- 位置编码(Positional Encoding)
- 层归一化(Layer Normalization)

### 1.2 自注意力机制

自注意力的计算公式如下:

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

其中:
- Q: 查询矩阵
- K: 键矩阵  
- V: 值矩阵
- d_k: 键向量的维度
- √d_k: 缩放因子,用于梯度的稳定性

### 1.3 多头注意力

多头注意力将输入分成多个头,每个头独立计算注意力后再合并:

1. 线性变换得到每个头的Q、K、V
2. 并行计算每个头的注意力分数
3. 拼接所有头的输出
4. 通过线性层映射回原始维度

### 1.4 位置编码

由于自注意力机制无法感知位置信息,需要额外添加位置编码。本实现使用可学习的位置嵌入。

### 1.5 自回归生成

GPT在生成时采用自回归方式:
1. 每次只预测下一个token
2. 将预测结果加入输入序列
3. 重复以上步骤直到生成结束符或达到最大长度

## 2. 代码结构

```
project/
├── data/                # 数据目录
│   ├── train.jsonl      # 原始训练数据
│   ├── train.json       # 处理后的训练数据
│   ├── val.json         # 验证数据
│   ├── identity_data.json # 身份设定数据
│   └── vocab.json       # 词表
├── output/              # 模型保存目录
│   ├── best.pt         # 最佳模型
│   └── last.pt         # 最新模型
├── logs/               # tensorboard日志
├── model.py            # 模型定义
├── tokenizer.py        # 分词器
├── qa_dataset.py       # 数据集
├── build_vocab.py      # 构建词表
├── train.py            # 训练脚本
└── predict.py          # 预测脚本
```

## 3. 快速开始

### 3.1 环境要求

需要 PyTorch >= 1.12.0
```bash
pip install torch numpy tqdm tensorboard
```

### 3.2 准备数据

数据格式为jsonl,每行包含问答对:
```json
{"question": "问题", "answer": "答案"}
```

推荐中文数据：[qiaojiedongfeng](https://modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/files)

可以通过修改 data/identity_data.json 来自定义模型的身份设定。

将数据文件放在data/train.jsonl,然后运行:
```bash
python split_dataset.py
```

### 3.3 构建词表

```bash
python build_vocab.py
```

### 3.4 训练模型

```bash
python train.py
```

可配置的主要参数:
- max_length: 序列最大长度
- batch_size: 批次大小
- lr: 学习率
- epochs: 训练轮数
- d_model: 模型维度
- n_heads: 注意力头数
- n_layers: 解码器层数

### 3.5 对话预测

```bash
python predict.py
```

## 4. 模型参数

默认配置:
- d_model: 768 (嵌入维度)
- d_ff: 2048 (前馈网络维度)
- n_heads: 8 (注意力头数)
- n_layers: 6 (解码器层数)
- max_length: 120 (最大序列长度)
- vocab_size: 4825 (词表大小)

总参数量: 37,128,409

## 5. 训练技巧

1. 梯度裁剪防止梯度爆炸
2. 使用Adam优化器
3. 学习率建议设置为1e-4
4. batch_size根据显存大小调整,建议128或256
5. 训练时使用teacher forcing
6. 使用交叉熵损失函数
7. 使用验证集监控过拟合

## 6. 注意事项

1. 设备支持优先级:
   - NVIDIA GPU (CUDA)
   - Apple Silicon (MPS)
   - CPU
2. 恢复训练时注意:
   - 如果在不同设备间迁移模型，需要使用正确的map_location
   - 确保所有模型参数和优化器状态都在同一设备上
3. 注意数据预处理的质量
4. 生成时注意控制最大长度
5. 模型体量较小,生成效果仅供学习参考

## 7. TODO

- [ ] 添加温度采样
- [ ] 实现beam search
- [ ] 支持模型量化
- [ ] 添加更多解码策略
- [ ] 支持模型断点续训
- [ ] 优化推理速度

## 8. 参考

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
3. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## License

MIT
