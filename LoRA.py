import torch.nn as nn
import torch
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.scale = alpha / rank
        self.W_a = nn.Parameter(torch.zeros(in_dim, rank), requires_grad=True)
        nn.init.kaiming_uniform_(self.W_a, a=math.sqrt(5))
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim), requires_grad=True)
    def forward(self, x):
        return self.scale * (x @ self.W_a @ self.W_b)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    def forward(self, x):
        return self.linear(x) + self.lora(x)



