```python3
torch.nn.parallel.DistributedDataParallel(
    module,
    device_ids=None,
    output_device=None,
    dim=0,
    broadcast_buffers=True,
    init_sync=True,
    process_group=None,
    bucket_cap_mb=None,
    find_unused_parameters=False,
    check_reduction=False,
    gradient_as_bucket_view=False,
    static_graph=False,
    delay_all_reduce_named_params=None,
    param_to_hook_all_reduce=None,
    mixed_precision: Optional[_MixedPrecision] = None,
    device_mesh=None,
    skip_all_reduce_unused_params=False,
)
```

## 参数

+ module

  要并行的模型。必须先 .to(device) 放到正确设备上

+ device_ids

  指定当前进程使用的 GPU id 列表。单 GPU 训练一般 [rank]

+ output_device

  输出张量放到哪块 GPU

+ dim

  对哪个维度进行梯度同步（梯度 bucket 分组）。通常默认即可

+ broadcast_buffers

  是否广播 BatchNorm / 其他 buffer（如 running_mean）

+ process_group

  指定分布式进程组，默认使用 ```dist.group.WORLD```

+ bucket_cap_mb

  梯度 bucket 大小（MB），梯度会分组通信。增大可以减少通信次数，但可能占用更多显存

+ find_unused_parameters

  模型中是否有部分参数在 forward 中未使用。如果有，需设为 True，DDP 会额外处理

+ check_reduction

  调试用，检查梯度同步是否完成

+ gradient_as_bucket_view

  提高性能的高级选项，让梯度视图直接在 bucket 内部修改

+ static_graph

  模型计算图固定时可设为 True，提高性能

+ delay_all_reduce_named_params

  
