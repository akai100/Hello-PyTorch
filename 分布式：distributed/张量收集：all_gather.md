```python3
def all_gather(tensor_list, tensor, group=None, async_op=False):
    ......
```

all_gather 的作用是：

+ 将 每个进程的张量 收集到 所有进程 中。

+ 最终每个进程都会得到一个 包含所有进程张量的列表 或 拼接后的张量。

+ 属于 全量通信（All-Reduce 类似），不同的是它不是求和，而是收集。

## 参数

+ tensor_list

  长度为 world_size 的张量列表，用于存放每个进程的张量。每个元素的形状必须和 tensor 相同。

+ tensor

  本地张量，即当前进程想要发送的数据

+ group

  通信组，默认是 torch.distributed.group.WORLD（即所有进程）

+ async_op

  否异步执行通信，返回 Work 对象，如果 True，需要 .wait() 才完成
