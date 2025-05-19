import torch
from torch.optim.optimizer import Optimizer
import math

class STORM(Optimizer):
    """STOchastic Recursive Momentum optimizer
    
    Args:
        params (iterable): Trainable parameters or parameter groups
        lr (float): Initial learning rate (default: 0.1)
        momentum (float): Momentum factor (default: 0.9)
        weight_decay (float): Weight decay (default: 0)
        c (float): Recursive momentum control parameter (default: 0.1)
        nesterov (bool): Whether to use Nesterov momentum (default: False)
        use_constant_c (bool): Whether to use constant c_t (default: False)
    """
    
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0, c=0.1, nesterov=False, use_constant_c=True):
        # Initialize counter
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
        
        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['momentum_buffer'] = torch.zeros_like(p.data)  # Recursive momentum
                state['prev_grad'] = torch.zeros_like(p.data)  # Previous step gradient
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update global step count
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
                
                # Get previous step state
                prev_momentum = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # Choose between constant c_t or decaying c_t
                if use_constant_c:
                    # Use constant c_t
                    c_t = c
                else:
                    # Use decaying c_t
                    epsilon = 1e-6 
                    c_t = c / math.sqrt(self.step_count + epsilon)
                
                # STORM recursive momentum update
                # d_t = (1-c_t) * d_{t-1} + g_t - (1-c_t) * g_{t-1}
                new_momentum = (1-c_t) * prev_momentum + grad - (1-c_t) * prev_grad
                
                # Save current gradient for next step
                state['prev_grad'] = grad.clone()
                state['momentum_buffer'] = new_momentum
                
                # Standard STORM update (without Nesterov acceleration)
                p.data.add_(new_momentum, alpha=-lr)
        
        return loss