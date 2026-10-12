

import torch.nn as nn

import random
import numpy as np
import torch
import os

# INIT
def kaiming_init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



def set_seed(seed: int = 42):
    """
    固定所有可能的随机来源，确保实验可复现
    """
    # Python 原生随机
    random.seed(seed)

    # Numpy 随机
    np.random.seed(seed)

    # PyTorch 随机
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU

    # CUDA 卷积算法确定性
    # 注意：这可能会略微降低运行速度，但能保证结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to: {seed} (Deterministic mode enabled)")




class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 提取层级规格
        in_sizes = [config['input']] + config['hidden']
        out_sizes = config['hidden'] + [config['output']]

        self.blocks = nn.ModuleList()
        self.use_residuals = []

        for i, (in_sz, out_sz, bias, drop, bn, ln, act) in enumerate(zip(
                in_sizes, out_sizes,
                config['use_biases'], config['dropouts'],
                config['use_batch_norms'], config['use_layer_norms'], config['activations']
        )):
            block = nn.ModuleDict()

            # 1. 核心线性层
            lin = nn.Linear(in_sz, out_sz, bias=bias)
            if act:
                act_lower = act.lower()
                if 'relu' in act_lower:
                    nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
                elif 'leaky_relu' in act_lower:
                    nn.init.kaiming_normal_(lin.weight, nonlinearity='leaky_relu')
                elif 'gelu' in act_lower:
                    nn.init.xavier_normal_(lin.weight)
                else:
                    nn.init.xavier_normal_(lin.weight)
            block['linear'] = lin

            # 2. 正则化层
            if bn: block['bn'] = nn.BatchNorm1d(out_sz)
            if ln: block['ln'] = nn.LayerNorm(out_sz)

            # 3. 激活函数
            if act:
                act_lower = act.lower()
                if act_lower == 'relu':
                    block['act'] = nn.ReLU()
                elif act_lower == 'tanh':
                    block['act'] = nn.Tanh()
                elif act_lower == 'gelu':
                    block['act'] = nn.GELU()
                elif act_lower.startswith("leaky_relu"):
                    slope = float(act.split(":")[1]) if ":" in act else 0.01
                    block['act'] = nn.LeakyReLU(slope)
                else:
                    raise ValueError(f"Unsupported activation: {act}")

            if drop > 0: block['dropout'] = nn.Dropout(drop)
            self.blocks.append(block)
            self.use_residuals.append(in_sz == out_sz and i < len(in_sizes) - 1)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            identity = x if self.use_residuals[i] else None
            x = block['linear'](x)

            if 'bn' in block: x = block['bn'](x)
            if 'ln' in block: x = block['ln'](x)
            if 'act' in block: x = block['act'](x)

            # 执行残差连接（使用标准加法，不改变原 identity 内存指针）
            if self.use_residuals[i]:
                x = x + identity

            if 'dropout' in block: x = block['dropout'](x)

        return x



class DEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_sizes = [config['input']] + config['hiddens']
        out_sizes = config['hiddens'] + [config['output']]

        self.layers = nn.ModuleList()

        # 记录每一层是否可以进行残差连接 (输入输出维度需一致)
        self.use_residual = []

        for i, (in_size, out_size, use_bias, dropout, use_bn, use_ln, activation) in enumerate(zip(
                in_sizes, out_sizes, config['use_biases'], config['dropouts'],
                config['use_batch_norms'], config['use_layer_norms'], config['activations']
        )):
            layer_block = nn.ModuleDict()

            # 1. 线性层
            lin = nn.Linear(in_size, out_size, bias=use_bias)
            # 优化初始化：回归任务建议使用 kaiming_normal
            nn.init.kaiming_normal_(lin.weight,
                                    nonlinearity='leaky_relu' if activation and 'leaky' in activation else 'relu')
            layer_block['linear'] = lin

            # 2. 正则化与激活
            if dropout > 0:
                layer_block['dropout'] = nn.Dropout(dropout)
            if use_bn:
                layer_block['bn'] = nn.BatchNorm1d(out_size)
            if use_ln:
                layer_block['ln'] = nn.LayerNorm(out_size)

            if activation is not None:
                if activation == 'relu':
                    layer_block['act'] = nn.ReLU()
                elif activation == 'leaky_relu' or activation.startswith("leaky_relu"):
                    slope = float(activation.split(":")[1]) if ":" in activation else 0.01
                    layer_block['act'] = nn.LeakyReLU(slope)
                elif activation == 'tanh':
                    layer_block['act'] = nn.Tanh()
                # 可以根据需要添加其他激活函数

            self.layers.append(layer_block)

            # 判断逻辑：如果输入输出维度相同且不是最后一层，开启残差
            # 最后一层通常不加残差，因为它是回归的目标输出
            is_res = (in_size == out_size) and (i < len(in_sizes) - 1)
            self.use_residual.append(is_res)

    def forward(self, x):
        for i, layer_block in enumerate(self.layers):
            identity = x  # 保存输入用于残差

            # 顺序：Linear -> Norm -> Activation -> Dropout
            x = layer_block['linear'](x)

            if 'bn' in layer_block: x = layer_block['bn'](x)
            if 'ln' in layer_block: x = layer_block['ln'](x)
            if 'act' in layer_block: x = layer_block['act'](x)

            # 应用残差连接
            if self.use_residual[i]:
                x = x + identity

            if 'dropout' in layer_block: x = layer_block['dropout'](x)

        return x
