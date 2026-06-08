from collections.abc import Callable
import math

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("learning rate must be greater than 0")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad.data
                state["t"] = t + 1
        return loss
    

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8, 
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be greater than 0")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
    

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr * weight_decay * p.data

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2

                p.data -= lr_t * m / (torch.sqrt(v) + eps) 

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss