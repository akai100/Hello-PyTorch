```torch.distributed.reduce``` 是 PyTorch 分布式通信库中用于**对多个进程上的张量进行归约（reduce）操作**的函数。它通常用于多卡训练或分布式训练场景下，将各个进程的张量汇总到指定的“主进程”上。

```python3
def reduce(
    tensor: torch.Tensor,
    dst: Optional[int] = None,
    op=ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_dst: Optional[int] = None,
)
```

## 参数

+ tensor

  要进行归约操作的张量（每个进程都有自己的张量）

+ dst

  目标进程的 rank（归约后的结果会在这个进程上）

+ op

  归约操作类型，默认是 ReduceOp.SUM，可选：SUM, PRODUCT, MIN, MAX, BAND 等

+ group

  通信组（默认为 group=dist.group.WORLD，即所有进程）

+ async_op

  是否异步执行，True 返回 Work 对象，可调用 .wait() 等待完成
