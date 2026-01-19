```torch.distributed.broadcast``` 是 PyTorch 分布式训练中最基础、最常用的通信原语之一，主要用于 把一个进程（rank）上的张量广播到同一进程组里的所有其他进程。

```python3
def broadcast(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_src: Optional[int] = None,
):
```

## 参数

+ tensor

  要广播的张量（所有 rank 都必须提供同 shape / dtype 的 tensor）

+ src

  源进程的 rank（数据来自这个 rank）

+ group

  进程组，默认是 WORLD

+ async_op

  是否异步执行，返回 Work 对象
