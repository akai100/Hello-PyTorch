```torch.distributed.all_reduce``` 是 PyTorch 分布式训练中常用的通信操作，用于对多个进程（通常在不同 GPU 或节点上）的张量执行 **全局归约（reduce）操作**，并将结果广播回每个进程。它常用于同步梯度或统计指标。

```python3
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    ......
```

## 参数

+ ```tensor```

  待归约的张量，必须在每个进程上存在

+ op

  归约操作，常用的有：

  + ```dist.ReduceOp.SUM```：求和
  + ```dist.ReduceOp.PRODUCT```：乘积
  + ```dist.ReduceOp.MIN```：最小值
  + ```dist.ReduceOp.MAX```：最大值

归约后的结果会 写回原 tensor，每个进程的 tensor 都会变成相同结果。


## 使用

## 使用场景

### 同步梯度

在分布式训练中，每个 GPU 会计算本地梯度，然后用 all_reduce 将梯度累加，确保每个 GPU 的梯度一致：

```python
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size  # 取平均梯度
```

## 注意事项
