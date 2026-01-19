```python3
def scatter(
    tensor: torch.Tensor,
    scatter_list: Optional[list[torch.Tensor]] = None,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_src: Optional[int] = None,
):
    ......
```

scatter 是 分布式通信原语（collective operation），用于 把一个张量列表（每个张量对应一个进程）从一个源进程发送到所有进程。

它的核心思想：**源进程持有 N 份数据，分别发送给 N 个进程。**

+ scatter 是 单向分发（不像 all_gather 是收集到每个进程）

+ 需要在分布式环境下调用（初始化了 init_process_group）

## 参数

+ scatter_list

  源进程（src）的张量列表，每个元素对应一个进程要接收的张量。非源进程传 None。

+ tensor

  当前进程接收的数据张量，大小必须和源进程对应张量一样

+ src

  数据源进程 rank。

+ group

  
