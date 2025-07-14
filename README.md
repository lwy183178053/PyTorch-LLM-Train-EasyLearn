# PyTorch-LLM-Train-EasyLearn
本项目致力于提供一个简洁易懂、适合初学者的 PyTorch 模型训练教程。从零实现多头注意力机制（MultiAttention）旋转位置编码（RoPE）等，便于快速掌握大语言模型（LLM）的算法实现与训练流程。

# 项目特点：  
✅ **零基础友好**：提供详细注释与分步教程，无需复杂背景知识  
✅ **模块化设计**：模型结构、训练流程与优化策略清晰分离，便于学习与修改  
无论是深度学习入门者希望理解Transformer核心机制，还是有经验的开发者需要快速实现实验原型，本项目都能为您提供清晰的代码示例与实用工具。  
💡 特别适合：  
- 希望系统学习PyTorch模型实现的学习者  
- 对Transformer相关技术感兴趣的研究者  
- 需要快速搭建自定义模型的工程实践者  

# 您可以通过本项目：  
1. 学习RoPE与RMSNorm的原理与实现细节  
2. 掌握PyTorch模型构建与训练的完整流程  
3. 基于现有框架快速实验自己的模型改进想法  
4. 获取可直接复用的代码模块用于其他项目  

# 使用教程
**1.安装依赖包**
- pip install transformers
- pip install modelscope
- pip install torch

**2.下载数据集**
- modelscope download --dataset 'gongjy/minimind_dataset' --local_dir './dataset/' sft_512.jsonl
- modelscope download --dataset 'gongjy/minimind_dataset' --local_dir './dataset/' pretrain_hq.jsonl
- modelscope download --dataset 'gongjy/minimind_dataset' --local_dir './dataset/' sft_mini_512.jsonl

**3.运行训练脚本**
- python train.py --train_class 0 # 预训练
- python train.py --train_class 1 # 微调

**4.运行tensorboard查看损失曲线**
- tensorboard --logdir=runs/0
- tensorboard --logdir=runs/1

