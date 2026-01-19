```python3
def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
    device_id: Optional[Union[torch.device, int]] = None,
) -> None:
```
作用：

+ 建立 进程之间的通信通道
+ 让各个进程知道：
  + 自己的 rank
  + world_size
+ 为后续操作提供基础：
  + ```DistributedDataParallel```
  + ```all_reduce / broadcast / gather```
  + ```DistributedSampler```

## 参数

+ backend
  
  通信后端，取值：

  + nccl：GPU 训练
  + gloo：CPU/少量 GPU
  + mpi：HPC 环境

+ world_size

  进程总数

  ```
  world_size = 机器数 × 每台机器 GPU 数
  ```

+ rank

  当前进程编号

  ```
  rank ∈ [0, world_size-1]
  ```

  + rank = 0：主进程（常用不 logging / save）
  + 每个进程 必须唯一

+ init_method

  进程发现方式

  最常用：
  
  ```
  init_method="env://"
  ```

  依赖以下环境变量：

  ```
  MASTER_ADDR
  MASTER_PORT
  WORLD_SIZE
  RANK
  ```
