```torch.utils.data.DataLoader``` 是 PyTorch 中用于批量加载数据的工具。它提供了一个可迭代的数据加载器，能够高效地从 Dataset 中加载数据并返回一个批次（batch），通常用于训练深度学习模型时。

```python3
DataLoader(
    dataset: Dataset[_T_co],
    batch_size: Optional[int] = 1,
    shuffle: Optional[bool] = None,
    sampler: Union[Sampler, Iterable, None] = None,
    batch_sampler: Union[Sampler[list], Iterable[list], None] = None,
    num_workers: int = 0,
    collate_fn: Optional[_collate_fn_t] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[_worker_init_fn_t] = None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
    in_order: bool = True,)
```

+ dataset

  传入一个继承自 torch.utils.data.Dataset 的对象。这个对象包含了如何从数据源加载单个数据点的代码。

+ batch_size

  每个批次加载的样本数。默认值是 1

+ shuffle

  是否打乱数据集。通常在训练时设置为 True，以确保每个epoch中的数据顺序不同，从而提升模型的泛化能力。

+ num_workers

  该参数控制用于加载数据的子进程数量。通过多进程来加速数据的读取，尤其在数据加载时间较长时。默认值是 0，表示不使用子进程。如果有多个 CPU 核心，可以增加 num_workers 来加速数据加载。

+ drop_last

  如果数据集的大小不能被 batch_size 整除，是否丢弃最后一个不完整的批次。默认是 False，即保留这个小批次。

+ pin_memory

  如果设置为 True，将数据加载到 CUDA 内存中（如果有 GPU）。这对于加速 GPU 上的训练非常有帮助。

+ collate_fn

  用于指定如何将多个数据样本合并成一个批次。默认的 collate_fn 会将数据样本堆叠成一个批次，但如果你需要特殊的合并方式，可以自己定义这个函数。

## 遍历 DataLoader

```python3
for batch_x, batch_y in dataloader:
    print(batch_x, batch_y)
```
