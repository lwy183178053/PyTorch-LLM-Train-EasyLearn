import os
import math
import time
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Subset
from torch import optim
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from model import MyModel
from dataset.lm_dataset import PretrainDataset, SFTDataset


# 余弦退火学习率
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 训练函数
def train(model, optimizer, criterion, train_loader, TensorBoard_Writer, config):
    # 混合精度训练
    scalar = torch.amp.GradScaler()
    num_epochs = config['num_epochs']
    print_step = config['print_step']
    save_step = config['save_step']
    device = config['device']
    model_save_path = config['model_save_path']

    for epoch in range(num_epochs):
        total_step = len(train_loader)
        total_time = 0
        total_loss = 0
        model.train()
        for step, (X, Y, attention_mask) in enumerate(train_loader):
            # 记录开始时间
            start_time = time.time()
            # 准备数据
            X = X.to(device)
            Y = Y.to(device)
            attention_mask = attention_mask.to(device)

            # 计算学习率
            lr = get_lr(epoch * total_step + step, num_epochs * total_step, config['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            optimizer.zero_grad()
            with torch.autocast(device_type=device):
                outputs = model(X, attention_mask)
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), Y.reshape(-1))

            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()

            # 记录结束时间
            end_time = time.time()
            total_time += end_time - start_time
            # 累加损失
            total_loss += loss.item()

            if step % print_step == 0 and step != 0:
                print(f'Epoch: {epoch} '
                      f'Step: {step}/{total_step} '
                      f'Lr: {lr:.2e} '
                      f'Loss: {total_loss / print_step:.4f} '
                      f'Time: {total_time:.2f}s')
                total_loss = 0
                total_time = 0

            if step % save_step == 0 and step != 0:
                # 保存模型
                torch.save(model.state_dict(), 'checkpoint.pth')
                print(f'Step {step}/{total_step} Model Save')

            if config['use_tensorboard']:
                # 添加 TensorBoard 记录
                TensorBoard_Writer.add_scalar('Loss/train', loss.item(), epoch * total_step + step)
                TensorBoard_Writer.add_scalar('Learning Rate', lr, epoch * total_step + step)

    # 训练结束后保存最终模型
    torch.save(model.state_dict(), 'checkpoint.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'{model_save_path} Model Save')


# 初始化模型、优化器、损失函数和数据加载器
def init_model(config):
    print(f'Device: {config["device"]}')
    print(f'Train Class: {config["train_class"]}')
    # 初始化 TensorBoard
    TensorBoard_Writer = SummaryWriter(log_dir=f'runs/{config["train_class"]}')
    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    print(f'Tokenizer Vocab Size: {tokenizer.vocab_size}')
    # 模型
    model = MyModel(vocab_size=tokenizer.vocab_size, embed_dim=config['embed_dim'],
                    num_layers=config['num_layers'], num_heads=config['num_heads']).to(config['device'])
    print(f'LLM Parameter Number: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    # 加载保存的模型权重
    if os.path.exists(config['model_path']):
        print(f'Load Model from {config["model_path"]}')
        model.load_state_dict(torch.load(config['model_path'], map_location=config['device'], weights_only=True))
    else:
        print(f'No Model Found at {config["model_path"]}, Train from Immediate parameters')
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # 数据集
    match config['train_class']:
        case 0:
            train_dataset = PretrainDataset(data_path=config['data_path'], tokenizer=tokenizer)
        case 1:
            train_dataset = SFTDataset(data_path=config['data_path'], tokenizer=tokenizer)
    # 选取前n个样本作为子集
    if config['subset']:
        train_dataset = Subset(train_dataset, indices=range(config['subset_size']))
    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    print(f'Train Batch Size: {len(train_loader)}')

    return model, optimizer, criterion, train_loader, TensorBoard_Writer


# 初始化配置
def init_config():
    config = {
        'train_class': 0,  # 训练类别,0为pretrain,1为sft
        'embed_dim': 768,
        'num_layers': 2,
        'num_heads': 8,
        'lr': 1e-4,
        'batch_size': 32,
        'num_epochs': 1,
        'shuffle': True,
        'subset': False,  # 是否使用子集
        'subset_size': 32000,  # 子集大小
        'use_tensorboard': True,  # 是否使用tensorboard
        'print_step': 100,  # 打印间隔
        'save_step': 500,  # 保存间隔
        'model_path': 'checkpoint.pth',
        'model_save_path': 'pre_model.pth',
        'data_path': 'dataset/sft_mini_512.jsonl',
        'tokenizer_path': 'tokenizer',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--train_class", type=int, help="训练类别,0为pretrain,1为sft", default=0)
    config["train_class"] = parser.parse_args().train_class

    if config["train_class"] == 0:
        config['data_path'] = 'dataset/pretrain_hq.jsonl'
        config['model_save_path'] = 'pre_model.pth'
    elif config["train_class"] == 1:
        config['data_path'] = 'dataset/sft_mini_512.jsonl'
        config['model_save_path'] = 'sft_mini_model.pth'

    return config

if __name__ == '__main__':
    # 初始化配置
    config = init_config()
    # 初始化模型、优化器、损失函数、数据加载器和TensorBoard
    model, optimizer, criterion, train_loader, TensorBoard_Writer = init_model(config)
    # 训练
    train(model, optimizer, criterion, train_loader, TensorBoard_Writer,config)
