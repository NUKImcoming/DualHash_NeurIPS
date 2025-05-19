import torch
from torch.optim.optimizer import Optimizer
import math

class STORM(Optimizer):
    """STOchastic Recursive Momentum优化器
    
    参数:
        params (iterable): 可训练参数或参数组
        lr (float): 初始学习率 (默认: 0.1)
        momentum (float): 动量因子 (默认: 0.9)
        weight_decay (float): 权重衰减 (默认: 0)
        c (float): 递归动量控制参数 (默认: 0.1)
        nesterov (bool): 是否使用Nesterov动量 (默认: False)
        use_constant_c (bool): 是否使用常数c_t (默认: False)
    """
    
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0, c=0.1, nesterov=False, use_constant_c=True):
        # 初始化计数
        self.step_count = 0
        self.use_constant_c = use_constant_c
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if c <= 0.0:
            raise ValueError(f"Invalid c value: {c}")
            
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, 
                        c=c, nesterov=nesterov, use_constant_c=use_constant_c)
        super(STORM, self).__init__(params, defaults)
        
        # 初始化状态
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['momentum_buffer'] = torch.zeros_like(p.data)  # 递归动量
                state['prev_grad'] = torch.zeros_like(p.data)  # 上一步梯度
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化
        
        参数:
            closure (callable, optional): 重新评估模型并返回损失的闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 更新全局步数
        self.step_count += 1
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            c = group['c']
            lr = group['lr']
            nesterov = group['nesterov']
            use_constant_c = group['use_constant_c'] if 'use_constant_c' in group else True
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                state['step'] += 1
                
                # 获取上一步的状态
                prev_momentum = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # 选择使用常数c_t还是衰减c_t
                if use_constant_c:
                    # 使用常数c_t
                    c_t = c
                else:
                    # 使用衰减c_t
                    epsilon = 1e-6 
                    c_t = c / math.sqrt(self.step_count + epsilon)
                
                # STORM递归动量更新
                # d_t = (1-c_t) * d_{t-1} + g_t - (1-c_t) * g_{t-1}
                new_momentum = (1-c_t) * prev_momentum + grad - (1-c_t) * prev_grad
                
                # 保存当前梯度用于下一步
                state['prev_grad'] = grad.clone()
                state['momentum_buffer'] = new_momentum
                
                # 标准STORM更新 (无Nesterov加速)
                p.data.add_(new_momentum, alpha=-lr)
        
        return loss