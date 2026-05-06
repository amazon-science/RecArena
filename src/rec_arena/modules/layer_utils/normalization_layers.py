import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization layer."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        output = output * self.weight + self.bias
        return output


class BatchNorm(nn.Module):
    """Batch normalization layer."""

    def __init__(self, d_model: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean = self.running_mean
            var = self.running_var
        output = (x - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output


class InstanceNorm(nn.Module):
    """Instance normalization layer."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        output = (x - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)
        return output


class GroupNorm(nn.Module):
    """Group normalization layer."""

    def __init__(self, num_groups: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        output = (x - mean) / torch.sqrt(var + self.eps)
        output = output.view(b, c, h, w)
        output = output * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output


class LearnableLayerScaling(nn.Module):
    """Learnable Layer Scaling (LLS) normalization layer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * self.weight.unsqueeze(0)
        return output


# RMSNorm is now provided by torch.nn.functional.rms_norm
# Use F.rms_norm(x, normalized_shape=[...], eps=...) instead
