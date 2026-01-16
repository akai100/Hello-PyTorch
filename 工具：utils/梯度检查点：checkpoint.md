**梯度检查点（Gradient Checkpointing）** 是减少显存的一种技术。在反向传播时，它会丢弃某些中间计算结果，只保存必要的计算图部分，避免了显存占用过多。

```python3
from torch.utils.checkpoint import checkpoint

def my_function(x):
    return model(x)

x = torch.randn(64, 1000, requires_grad=True).to('cuda')
output = checkpoint(my_function, x)

```
